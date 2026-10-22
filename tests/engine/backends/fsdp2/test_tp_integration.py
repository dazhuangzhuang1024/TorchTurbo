"""Integrated dense TP/FSDP2/Ulysses training on CPU/Gloo.

The CPU attention bridge requests the registered maximum-logit provider and
discards maxima. It does not add a production CPU attention implementation.
The oracle uses independent, per-segment SDPA on a complete unsharded model.
"""

import copy
import json
import os
import sys
from datetime import timedelta
from itertools import product
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.distributed.tensor import DTensor
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from tests.engine.backends.fsdp2.test_tensor_parallel import (
    DecoderBlock, DenseDecoder, HeadAttention, _native_plan,
)
from torchturbo.engine import TorchTurboEngine
from torchturbo.engine.checkpoint.coordinator import CheckpointCoordinator
from torchturbo.engine.checkpoint.metadata import load_metadata
from torchturbo.engine.config import EngineConfig
from torchturbo.engine.optimizers import OptimizerClassPlan
from torchturbo.models.base import TurboModel
from torchturbo.parallel import FSDPPlanSpec


_MAX_ERRORS = {}


class PackedHeadAttention(HeadAttention):
    """Extend the shared decoder fixture with packed attention semantics."""

    def __init__(self):
        super().__init__()
        # The native CPU provider implements equal Q/K/V head counts.
        self.num_key_value_heads = self.num_heads
        self.k_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.sp_info = None

    def forward(self, hidden, cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k):
        shape = hidden.shape[:2]
        query = self.q_norm(self.q_proj(hidden).view(*shape, self.num_heads, 2))
        key = self.k_norm(self.k_proj(hidden).view(*shape, self.num_key_value_heads, 2))
        value = self.v_proj(hidden).view(*shape, self.num_key_value_heads, 2)
        if self.sp_info is None:
            segments = []
            boundaries = cu_seq_lens_q.tolist()
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                result = F.scaled_dot_product_attention(
                    query[:, start:end].transpose(1, 2),
                    key[:, start:end].transpose(1, 2),
                    value[:, start:end].transpose(1, 2),
                    is_causal=True,
                )
                segments.append(result.transpose(1, 2))
            segments.append(torch.zeros_like(value[:, boundaries[-1]:]))
            attention = torch.cat(segments, dim=1)
        else:
            from torchturbo.models.modules.attention.ulysses.ulysses import (
                ulysses_flash_attention_forward,
            )

            attention = ulysses_flash_attention_forward(
                query, key, value, None, shape[1],
                sp_group=self.sp_info["sp_group"], sp_size=self.sp_info["sp_size"],
                is_causal=True, dropout=0.0, softmax_scale=2 ** -0.5,
                cu_seq_lens_q=cu_seq_lens_q, cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q, max_length_k=max_length_k,
            )
        return self.o_proj(attention.reshape(*shape, self.num_heads * 2))


class PackedDecoderBlock(DecoderBlock):
    def __init__(self):
        super().__init__()
        self.attn = PackedHeadAttention()

    def forward(self, hidden, **packed):
        hidden = hidden + self.attn(self.norm(hidden), **packed)
        value = self.ffn_norm(hidden)
        return hidden + self.down(F.silu(self.gate(value)) * self.up(value))


class PackedDenseDecoder(DenseDecoder, TurboModel):
    """Shared native TP fixture with the engine's packed causal-LM contract."""

    def __init__(self, tied=False):
        super().__init__(tied=tied)
        self.layers = nn.ModuleList([PackedDecoderBlock()])
        self.config = PretrainedConfig(tie_word_embeddings=tied)
        self.all_tied_weights_keys = {"head.weight": "embed.weight"} if tied else {}
        self.prepare_calls = 0
        self.input_activations = []
        self.last_hidden = None
        with torch.no_grad():
            self.layers[0].attn.q_norm.weight.copy_(torch.tensor([0.7, 1.3]))
            self.layers[0].attn.k_norm.weight.copy_(torch.tensor([1.1, 0.9]))

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self.head.weight = self.embed.weight

    def on_prepare_inputs(self, inputs, *, source_inputs=None):
        self.prepare_calls += 1
        super().on_prepare_inputs(inputs, source_inputs=source_inputs)
        prepared = dict(inputs)
        prepared["labels"] = inputs["labels"].clone()
        # Make hook ordering observable in both exact counts and the objective.
        if prepared["labels"][0, 4] != -100:
            prepared["labels"][0, 4] = -100
        return prepared

    def forward(self, input_ids, cu_seq_lens_q, cu_seq_lens_k,
                max_length_q, max_length_k, **kwargs):
        hidden = self.embed(input_ids)
        if torch.is_grad_enabled():
            hidden.retain_grad()
            self.input_activations.append(hidden)
        packed = dict(cu_seq_lens_q=cu_seq_lens_q, cu_seq_lens_k=cu_seq_lens_k,
                      max_length_q=max_length_q, max_length_k=max_length_k)
        for block in self.layers:
            hidden = block(hidden, **packed)
        hidden = self.norm(hidden)
        self.last_hidden = hidden.detach().clone()
        return CausalLMOutput(logits=self.head(hidden))


def _batch(data_rank, micro, step):
    # Valid packed segments are [0,1), [1,3), [3,6), followed by a physical
    # tail. SP=2 adds one more alignment token to this seven-token row.
    ids = (torch.arange(7).unsqueeze(0) + 1 + data_rank * 7 + step + micro) % 24
    labels = ids.clone()
    if micro == 1 or (data_rank == 1 and micro == 0):
        labels.fill_(-100)
    elif data_rank == 1:
        labels[0, 5] = -100
    cu = torch.tensor([0, 1, 3, 6], dtype=torch.int32)
    return dict(input_ids=ids, labels=labels, cu_seq_lens_q=cu,
                cu_seq_lens_k=cu.clone(), max_length_q=3, max_length_k=3,
                position_ids=torch.tensor([[0, 0, 1, 0, 1, 2, 0]]))


def _source(directory, tied):
    torch.manual_seed(913)
    model = PackedDenseDecoder(tied=tied)
    directory.mkdir()
    state = {name: value.detach().clone() for name, value in model.named_parameters()}
    save_file(state, str(directory / "model.safetensors"))
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(value.numel() * 4 for value in state.values())},
        "weight_map": {name: "model.safetensors" for name in state},
    }))
    model.config.save_pretrained(directory)


def _full(value):
    if isinstance(value, DTensor):
        value = value.full_tensor()
    return value.detach().clone()


def _assert_close(actual, expected, name):
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6, msg=name)
    category = name
    for prefix in ("clipped gradient", "gradient", "optimizer", "parameter"):
        if name.startswith(f"{prefix} "):
            category = prefix
            break
    difference = float((actual.detach() - expected.detach()).abs().max())
    _MAX_ERRORS[category] = max(_MAX_ERRORS.get(category, 0.0), difference)


def _prepare(source, sp_size, chunked, checkpointed, tied, *,
             mixed_precision="fp32", use_reentrant=False, optimizer_eps=1e-6):
    config = EngineConfig.from_dict({"mixed_precision": mixed_precision,
                                     "sp_size": sp_size, "tp_size": 2})
    config.loss_chunk.enabled = chunked
    config.loss_chunk.chunk_size = 2
    config.activation_checkpointing.enabled = checkpointed
    config.activation_checkpointing.use_reentrant = use_reentrant
    config.mfu.enabled = False
    engine = TorchTurboEngine(config)
    with torch.device("meta"):
        model = PackedDenseDecoder(tied=tied)
    plan = _native_plan()
    plan.sp_size = sp_size
    plan.plan["layers.*"] = FSDPPlanSpec(activation_checkpoint=checkpointed)
    engine.backend.prepare_model(model, plan)
    context = engine.backend.get_parallel_context()
    for block in model.layers:
        block.attn.sp_info = context["sp"]
    coordinator = CheckpointCoordinator(config, engine.backend.create_checkpoint_manager(),
                                        parallel_context=context)
    old_parameters = list(model.parameters())
    old_ids = {id(parameter) for parameter in old_parameters}
    coordinator.load_pretrained(model, str(source))
    engine.backend.finalize_model_load(model)
    assert not old_ids.intersection(id(parameter) for parameter in model.parameters())
    assert all(not parameter.is_meta for parameter in model.parameters())
    assert all(not engine.backend._parameter_ownership.is_expert(parameter)
               for parameter in model.parameters())
    assert engine.backend._parameter_ownership.expert_fqns(model) == frozenset()
    if tied:
        assert model.head.weight is model.embed.weight
    engine.model = model
    engine.parallel_plan = plan
    engine.model_name_or_path = str(source)
    engine._checkpoint = coordinator
    optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=.003,
                                         betas=(.8, .95), weight_decay=.03,
                                         eps=optimizer_eps, foreach=True, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
    assert len(optimizer.param_groups) > 2
    for group in optimizer.param_groups:
        layouts = {(parameter.device_mesh, parameter.placements) for parameter in group["params"]}
        assert len(layouts) == 1
    return engine, model, optimizer, scheduler


def _reference(source, tied, *, optimizer_eps=1e-6):
    model = PackedDenseDecoder(tied=tied)
    state = load_file(str(source / "model.safetensors"))
    if tied:
        state["head.weight"] = state["embed.weight"]
    model.load_state_dict(state)
    plan = OptimizerClassPlan(model, torch.optim.AdamW, {"lr": .003, "weight_decay": .03})
    groups = [group.get_optimizer_param_group() for group in plan.get_param_groups()]
    optimizer = torch.optim.AdamW(groups, lr=.003, betas=(.8, .95),
                                  weight_decay=.03, eps=optimizer_eps, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
    return model, optimizer, scheduler


def _oracle_step(model, step, micros):
    losses, batches, logits = [], {}, {}
    count = 0
    for data_rank in range(2):
        for micro in micros:
            batch = _batch(data_rank, micro, step)
            # Independently reproduce the fixture hook's declared transform.
            batch["labels"][0, 4] = -100
            output = model(**batch).logits
            batches[data_rank, micro] = model.input_activations[-1]
            logits[data_rank, micro] = output.detach()
            # Literal boundaries avoid sharing production label preparation.
            for start, end in ((0, 1), (1, 3), (3, 6)):
                targets = batch["labels"][:, start + 1:end]
                count += int((targets != -100).sum())
                losses.append(F.cross_entropy(output[:, start:end - 1].reshape(-1, 24),
                                              targets.reshape(-1), ignore_index=-100,
                                              reduction="sum"))
    loss = sum(losses) / count
    loss.backward()
    return count, loss.detach(), batches, logits


def _assert_optimizer(model, optimizer, reference, reference_optimizer):
    for name, parameter in model.named_parameters():
        expected = reference.get_parameter(name)
        _assert_close(_full(parameter), expected, f"parameter {name}")
        state, expected_state = optimizer.state[parameter], reference_optimizer.state[expected]
        assert state.keys() == expected_state.keys()
        for key in state:
            _assert_close(_full(state[key]), expected_state[key], f"optimizer {name} {key}")


def _train_step(engine, model, optimizer, scheduler, reference,
                reference_optimizer, reference_scheduler, step, micros, clipping):
    optimizer.zero_grad(set_to_none=True)
    reference_optimizer.zero_grad(set_to_none=True)
    model.input_activations.clear()
    reference.input_activations.clear()
    count, expected_loss, expected_inputs, expected_logits = _oracle_step(reference, step, micros)
    context = engine.backend._parallel_context
    data_rank = context.get_dp_info()["dp_rank"]
    sp_rank = context.get_sp_info()["sp_rank"]
    sources = [_batch(data_rank, micro, step) for micro in micros]
    snapshots = [{name: value.clone() if isinstance(value, torch.Tensor) else value
                  for name, value in batch.items()} for batch in sources]
    before_calls = model.prepare_calls
    prepared, actual_count = engine.prepare_loss_step(model, sources, torch.device("cpu"))
    assert actual_count == count
    assert model.prepare_calls == before_calls + len(micros)
    report = 0
    for micro, batch in zip(micros, prepared):
        assert all(value.device.type == "cpu" for value in batch.values()
                   if isinstance(value, torch.Tensor))
        loss, output = engine.compute_prepared_loss(
            model, batch, True, num_items_in_batch=count,
            gradient_accumulation_steps=len(micros),
        )
        assert not output.main_loss.requires_grad
        report = report + output.main_loss
        actual_logits = output.logits
        if actual_logits is None:
            actual_logits = F.linear(model.last_hidden, _full(model.head.weight))
        local_length = actual_logits.shape[1]
        start = sp_rank * local_length
        valid_end = min(start + local_length, 6)
        _assert_close(actual_logits[:, :valid_end - start],
                      expected_logits[data_rank, micro][:, start:valid_end], "valid logits")
        loss.backward()
        assert "labels" in batch and "cu_seq_lens_q" in batch
        activation = model.input_activations[-1]
        expected = expected_inputs[data_rank, micro].grad
        expected = F.pad(expected, (0, 0, 0, (-expected.shape[1]) % context.sp_size))
        expected = expected[:, start:start + local_length] * context.dense_fsdp_mesh.size()
        _assert_close(activation.grad, expected, "decoder input gradient = K * logical gradient")
    assert model.prepare_calls == before_calls + len(micros)
    for source, snapshot in zip(sources, snapshots):
        assert source.keys() == snapshot.keys()
        for name in source:
            if isinstance(source[name], torch.Tensor):
                torch.testing.assert_close(source[name], snapshot[name], rtol=0, atol=0)
            else:
                assert source[name] == snapshot[name]
    _assert_close(report, expected_loss, "optimizer-step report")
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        _assert_close(_full(parameter.grad), reference.get_parameter(name).grad, f"gradient {name}")
    expected_norm = torch.linalg.vector_norm(torch.cat([
        parameter.grad.flatten() for parameter in reference.parameters()
    ]))
    actual_norm = engine.backend.clip_grad_norm_(model.parameters(), float("inf"))
    _assert_close(actual_norm, expected_norm, "global gradient norm")
    if clipping:
        assert expected_norm > .15
        actual_norm = engine.backend.clip_grad_norm_(model.parameters(), .15)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), .15)
        _assert_close(actual_norm, expected_norm, "pre-clipping norm")
    for name, parameter in model.named_parameters():
        _assert_close(_full(parameter.grad), reference.get_parameter(name).grad, f"clipped gradient {name}")
    optimizer.step()
    reference_optimizer.step()
    scheduler.step()
    reference_scheduler.step()
    _assert_optimizer(model, optimizer, reference, reference_optimizer)
    scheduler_state = scheduler.state_dict()
    reference_scheduler_state = reference_scheduler.state_dict()
    # Native layout bucketing splits the same semantic groups into more
    # optimizer groups, so scheduler LR lists have different cardinalities.
    for key in scheduler_state:
        if key in ("base_lrs", "_last_lr"):
            assert set(scheduler_state[key]) == set(reference_scheduler_state[key])
        else:
            assert scheduler_state[key] == reference_scheduler_state[key]
    return float(report), count


def _assert_native_equal(model, optimizer, restored, restored_optimizer):
    for name, parameter in model.named_parameters():
        other = restored.get_parameter(name)
        torch.testing.assert_close(_full(parameter), _full(other), rtol=0, atol=0)
        for key in optimizer.state[parameter]:
            torch.testing.assert_close(_full(optimizer.state[parameter][key]),
                                       _full(restored_optimizer.state[other][key]), rtol=0, atol=0)


def _checkpoint_roundtrip(engine, model, optimizer, scheduler, reference,
                          reference_optimizer, reference_scheduler, directory,
                          source, sp_size, chunked, checkpointed, tied, clipping):
    # Data cursor and step are owned by the caller, not the TT coordinator.
    caller_state = {"completed_steps": 2, "next_microbatches": [1, 2]}
    if dist.get_rank() == 0:
        directory.mkdir()
        (directory / "trainer_state.json").write_text(json.dumps(caller_state))
    dist.barrier()
    engine.save_model(model, str(directory), final=True)
    engine.save_optimizer_and_scheduler(model, optimizer, scheduler, str(directory))
    restored_engine, restored, restored_optimizer, restored_scheduler = _prepare(
        source, sp_size, chunked, checkpointed, tied,
    )
    parameter_ids = [id(parameter) for parameter in restored.parameters()]
    restored_engine.load_from_checkpoint(restored, str(directory))
    restored_engine.load_optimizer_and_scheduler(restored, restored_optimizer,
                                                 restored_scheduler, str(directory))
    assert parameter_ids == [id(parameter) for parameter in restored.parameters()]
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    _assert_native_equal(model, optimizer, restored, restored_optimizer)
    assert load_metadata(str(directory))["schema_version"] == 2
    exported = {}
    hf_directory = directory / "huggingface"
    index = json.loads((hf_directory / "model.safetensors.index.json").read_text())
    for filename in set(index["weight_map"].values()):
        exported.update(load_file(str(hf_directory / filename)))
    expected_names = {name for name, _ in model.named_parameters()}
    assert set(exported) == expected_names
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(exported[name], _full(parameter), rtol=0, atol=0)
    reloaded_caller = json.loads((directory / "trainer_state.json").read_text())
    assert reloaded_caller == caller_state
    restored_reference, restored_reference_optimizer, restored_reference_scheduler = _reference(source, tied)
    restored_reference.load_state_dict(reference.state_dict())
    restored_reference_optimizer.load_state_dict(copy.deepcopy(reference_optimizer.state_dict()))
    restored_reference_scheduler.load_state_dict(reference_scheduler.state_dict())
    for objects in (
        (engine, model, optimizer, scheduler, reference, reference_optimizer, reference_scheduler),
        (restored_engine, restored, restored_optimizer, restored_scheduler,
         restored_reference, restored_reference_optimizer, restored_reference_scheduler),
    ):
        _train_step(*objects, step=reloaded_caller["completed_steps"],
                    micros=reloaded_caller["next_microbatches"], clipping=clipping)
    _assert_native_equal(model, optimizer, restored, restored_optimizer)
    assert scheduler.state_dict() == restored_scheduler.state_dict()
    if tied:
        assert restored.head.weight is restored.embed.weight


def _worker(rank, rendezvous, directory, sp_size):
    from torchturbo.models.modules.attention.ulysses import ulysses
    from torchturbo.models.modules.attention.interface import attention_forward

    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=4 * sp_size, timeout=timedelta(seconds=120))
    bridge_calls = []
    _MAX_ERRORS.clear()

    def cpu_attention_bridge(*values, **options):
        assert not options.pop("return_max_logits", False)
        bridge_calls.append(tuple(values[0].shape))
        output, _ = attention_forward(*values, return_max_logits=True, **options)
        return output

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ulysses, "attention_forward", cpu_attention_bridge)
            for chunked, checkpointed, clipping in product((False, True), repeat=3):
                cases = (False, True) if not chunked and checkpointed and clipping else (False,)
                for tied in cases:
                    name = f"s{sp_size}-chunk{chunked}-ac{checkpointed}-clip{clipping}-tied{tied}"
                    case_dir = Path(directory) / name
                    source = case_dir / "source"
                    if rank == 0:
                        print(f"INTEGRATED CASE START {name}", flush=True)
                        case_dir.mkdir()
                        _source(source, tied)
                    dist.barrier()
                    engine, model, optimizer, scheduler = _prepare(
                        source, sp_size, chunked, checkpointed, tied,
                    )
                    reference, reference_optimizer, reference_scheduler = _reference(source, tied)
                    for parameter_name, parameter in model.named_parameters():
                        torch.testing.assert_close(_full(parameter), reference.get_parameter(parameter_name),
                                                   rtol=0, atol=0)
                    bridge_start = len(bridge_calls)
                    for step, micros in enumerate(((0, 1, 2), (2,))):
                        report, count = _train_step(
                            engine, model, optimizer, scheduler, reference,
                            reference_optimizer, reference_scheduler, step, micros, clipping,
                        )
                        assert count == (5 if step == 0 else 3)
                    num_calls = len(bridge_calls) - bridge_start
                    assert num_calls == 4 * (2 if checkpointed else 1)
                    assert all(shape[2] == 8 // (2 * sp_size)
                               for shape in bridge_calls[bridge_start:])
                    if checkpointed and clipping:
                        # The smaller and larger meshes both resume the exact
                        # integrated loss, norm and optimizer path.
                        _checkpoint_roundtrip(
                            engine, model, optimizer, scheduler, reference,
                            reference_optimizer, reference_scheduler, case_dir / "checkpoint",
                            source, sp_size, chunked, checkpointed, tied, clipping,
                        )
                    if rank == 0:
                        print(f"INTEGRATED CASE PASS {name}: report={report:.8f} count={count}", flush=True)
            rank_errors = [None] * dist.get_world_size()
            dist.all_gather_object(rank_errors, _MAX_ERRORS)
            if rank == 0:
                errors = {name: max(values[name] for values in rank_errors) for name in _MAX_ERRORS}
                print(f"INTEGRATED MAX ABS ERRORS S{sp_size}: {json.dumps(errors, sort_keys=True)}", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("sp_size", [1, 2], ids=["D2-S1-T2", "D2-S2-T2"])
def test_dense_tp_fsdp_ulysses_training_and_resume(tmp_path, sp_size):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), sp_size),
             nprocs=4 * sp_size, join=True)
