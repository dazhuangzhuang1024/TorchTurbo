"""Native sparse-Adam and HF continuation for the two accepted Bailing rows."""

import copy
import json
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.distributed.tensor import DTensor

from tests.engine.backends.fsdp2.etp_checkpoint_helpers import (
    assert_tensor_bytes_equal, read_hf,
)
from tests.engine.backends.fsdp2.test_tp_checkpoint import _assert_state_equal
from tests.models.qwen3_moe.etp_helpers import full
from .etp_helpers import (
    LANGUAGE_CASES, BailingCase, BailingEvidence, bailing_batch, converted_fields,
    observe, prepare_bailing, reference_model, train_step, write_bailing_source,
)
from .tp_helpers import cpu_bailing_leaves


def assert_exact_sparse_state(left, left_optimizer, right, right_optimizer):
    """Compare complete model bytes and only the natively observed Adam slots."""
    _assert_state_equal(left.state_dict(), right.state_dict())
    _assert_state_equal(left_optimizer.state_dict(), right_optimizer.state_dict())
    for name, value in left.state_dict().items():
        assert_tensor_bytes_equal(value, right.state_dict()[name])
    left_slots = {name for name, p in left.named_parameters() if p in left_optimizer.state}
    right_slots = {name for name, p in right.named_parameters() if p in right_optimizer.state}
    assert left_slots == right_slots
    for name in sorted(left_slots):
        slots = left_optimizer.state[left.get_parameter(name)]
        other = right_optimizer.state[right.get_parameter(name)]
        assert slots.keys() == other.keys()
        for field, value in slots.items():
            assert_tensor_bytes_equal(value, other[field])


def record_native_state(directory, label, model, optimizer, scheduler):
    """Retain logical/local tensor values without initializing optimizer slots."""
    def tensor(value):
        return dict(full=full(value), local=value.to_local().detach().clone()
                    if isinstance(value, DTensor) else value.detach().clone())
    names = {p: name for name, p in model.named_parameters()}
    state = dict(model={name: tensor(value) for name, value in model.state_dict().items()},
        optimizer={names[p]: {field: tensor(value) for field, value in slot.items()}
                   for p, slot in optimizer.state.items()},
        groups=[{**{key: value for key, value in group.items() if key != "params"},
                 "params": [names[p] for p in group["params"]]} for group in optimizer.param_groups],
        scheduler=scheduler.state_dict())
    torch.save(state, directory / f"rank-{dist.get_rank()}-{label}.pt")


def assert_variant_schema(values, case):
    mla, kda = "model.layers.1.attention.", "model.layers.0.attention."
    required = [mla + key for key in ("g_proj.weight", "kv_a_proj_with_mqa.weight",
        "kv_a_layernorm.weight", "kv_b_proj.weight", "dense.weight")]
    required += [kda + key for key in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
        "A_log", "dt_bias", "o_norm.weight", "o_proj.weight")]
    if case.family == "hybrid":
        required += [mla + key for key in ("q_a_proj.weight", "q_b_proj.weight", "q_a_layernorm.weight")]
        required += [kda + key + ".weight" for key in ("f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj")]
        absent = [mla + "q_proj.weight", kda + "f_proj.weight", kda + "g_proj.weight"]
        assert values[mla + "g_proj.weight"].shape == (4, 16)
    else:
        required += [mla + "q_proj.weight", kda + "f_proj.weight", kda + "g_proj.weight"]
        absent = [mla + key for key in ("q_a_proj.weight", "q_b_proj.weight", "q_a_layernorm.weight")]
        absent += [kda + key + ".weight" for key in ("f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj")]
        required += ["model.visual.patch_embed.proj.weight", "linear_proj.0.weight", "linear_proj.2.weight"]
        assert not any("language_model" in name for name in values)
        assert values[mla + "g_proj.weight"].shape == (16, 16)
    required += ["model.layers.1.mlp.gate.expert_bias"]
    for prefix in ("model.layers.0.mlp.", "model.layers.1.mlp.shared_experts."):
        required += [prefix + projection + ".weight" for projection in ("gate_proj", "up_proj", "down_proj")]
    for expert in range(8):
        required += [f"model.layers.1.mlp.experts.{expert}.{projection}.weight"
                     for projection in ("gate_proj", "up_proj", "down_proj")]
    assert set(required) <= values.keys()
    assert not set(absent).intersection(values)


def warm_forward(engine, model, case):
    context = engine.backend._parallel_context
    batches = [bailing_batch(context.data_dp_mesh.get_local_rank(), 0, 1, case)]
    prepared, count = engine.prepare_loss_step(model, batches, torch.device("cpu"))
    stages, _, _, handles = observe(model)
    model.eval()
    try:
        with torch.no_grad():
            loss, output = engine.compute_prepared_loss(model, {**prepared[0], "use_cache": False},
                                                       True, num_items_in_batch=count)
            logits = output.logits
            if logits is None:
                logits = F.linear(stages["final"], full(model.lm_head.weight))
            return loss.detach().clone(), logits.detach().clone()
    finally:
        model.train()
        for handle in handles:
            handle.remove()


def checkpoint_worker(rank: int, directory: str, case: BailingCase) -> None:
    directory = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank,
                            world_size=case.world, timeout=timedelta(seconds=120))
    evidence, events = BailingEvidence(), []
    result = dict(rank=rank, case=asdict(case), complete=False)
    try:
        with cpu_bailing_leaves(events):
            source, checkpoint = directory / "source", directory / "checkpoint"
            engine, model, optimizer, scheduler = prepare_bailing(source, case)
            reference, ref_optimizer, ref_scheduler = reference_model(source, case)
            identities = {name: id(p) for name, p in model.named_parameters()}
            optimizer_ids = [[id(p) for p in group["params"]] for group in optimizer.param_groups]
            bias = full(model.model.layers[1].mlp.router.expert_bias)
            train_step(engine, model, optimizer, scheduler, reference, ref_optimizer,
                       ref_scheduler, case, 0, 2, evidence)
            record_native_state(directory, "saved-step1", model, optimizer, scheduler)
            saved_slots = [name for name, p in model.named_parameters() if p in optimizer.state]
            unused = [name for name, p in model.named_parameters()
                      if ".visual." in name or name.startswith("linear_proj.")]
            for name in unused:
                p = model.get_parameter(name)
                assert p.requires_grad and p.grad is None and p not in optimizer.state
            if case.family == "vl":
                assert len(unused) == 21
            engine.save_model(model, str(checkpoint), final=True)
            engine.save_optimizer_and_scheduler(model, optimizer, scheduler, str(checkpoint))
            assert saved_slots == [name for name, p in model.named_parameters() if p in optimizer.state]
            assert identities == {name: id(p) for name, p in model.named_parameters()}
            assert optimizer_ids == [[id(p) for p in group["params"]] for group in optimizer.param_groups]

            # The target is freshly prepared; only real DCP supplies its state.
            resumed_engine, resumed, resumed_optimizer, resumed_scheduler = prepare_bailing(source, case)
            target_ids = {name: id(p) for name, p in resumed.named_parameters()}
            target_optimizer_ids = [[id(p) for p in group["params"]] for group in resumed_optimizer.param_groups]
            resumed_engine.load_from_checkpoint(resumed, str(checkpoint))
            resumed_engine.load_optimizer_and_scheduler(resumed, resumed_optimizer, resumed_scheduler, str(checkpoint))
            assert target_ids == {name: id(p) for name, p in resumed.named_parameters()}
            assert target_optimizer_ids == [[id(p) for p in group["params"]] for group in resumed_optimizer.param_groups]
            assert_exact_sparse_state(model, optimizer, resumed, resumed_optimizer)
            _assert_state_equal(scheduler.state_dict(), resumed_scheduler.state_dict())
            result["native_restored_bytes_equal"] = True
            record_native_state(directory, "restored-step1", resumed, resumed_optimizer, resumed_scheduler)

            exported, original = read_hf(checkpoint / "huggingface"), read_hf(source)
            assert exported.keys() == original.keys() == reference.state_dict().keys()
            assert_variant_schema(original, case)
            assert_variant_schema(exported, case)
            for name, value in exported.items():
                assert value.shape == original[name].shape
                evidence.close(value, reference.state_dict()[name], "HF oracle " + name)
            converter = model.get_weight_converter(engine.backend.get_parallel_context(), str(source))
            for name, value in model.state_dict().items():
                for hf_name, wanted in converted_fields(converter, name, value).items():
                    assert_tensor_bytes_equal(exported[hf_name], wanted)
            source_config = json.loads((source / "config.json").read_text())
            hf_config = json.loads((checkpoint / "huggingface" / "config.json").read_text())
            assert source_config == hf_config
            text = hf_config if case.family == "hybrid" else hf_config["text_config"]
            assert text["use_cache"] is True
            assert text["num_attention_heads"] == 4 and text["num_key_value_heads"] == 1
            assert not any(key in hf_config or key in text for key in ("tp_size", "etp_size", "sp_size", "ep_size", "tp_mesh"))
            hf_engine, hf_model, hf_optimizer, _ = prepare_bailing(checkpoint / "huggingface", case)
            assert hf_model.model.config.use_cache is False
            for name, value in model.state_dict().items():
                assert_tensor_bytes_equal(value, hf_model.state_dict()[name])
            warm, imported_warm = warm_forward(engine, model, case), warm_forward(hf_engine, hf_model, case)
            for actual, expected in zip(warm, imported_warm):
                assert_tensor_bytes_equal(actual, expected)
            assert not hf_optimizer.state
            result.update(hf_keys=len(exported), source_cache=True, runtime_cache=False,
                          hf_model_bytes_equal=True, hf_warm_max_error=0.)

            resumed_reference, resumed_ref_optimizer, resumed_ref_scheduler = reference_model(source, case)
            resumed_reference.load_state_dict(reference.state_dict())
            resumed_ref_optimizer.load_state_dict(copy.deepcopy(ref_optimizer.state_dict()))
            resumed_ref_scheduler.load_state_dict(ref_scheduler.state_dict())
            for label, objects in zip(("uninterrupted-step2", "resumed-step2"),
                           ((engine, model, optimizer, scheduler, reference, ref_optimizer, ref_scheduler),
                            (resumed_engine, resumed, resumed_optimizer, resumed_scheduler,
                             resumed_reference, resumed_ref_optimizer, resumed_ref_scheduler))):
                train_step(*objects, case, 1, 1, evidence)
                record_native_state(directory, label, *objects[1:4])
            assert_exact_sparse_state(model, optimizer, resumed, resumed_optimizer)
            _assert_state_equal(scheduler.state_dict(), resumed_scheduler.state_dict())
            assert evidence.steps[-2]["reported_ce"] == evidence.steps[-1]["reported_ce"]
            assert target_ids == {name: id(p) for name, p in resumed.named_parameters()}
            assert target_optimizer_ids == [[id(p) for p in group["params"]] for group in resumed_optimizer.param_groups]
            for current, current_optimizer in ((model, optimizer), (resumed, resumed_optimizer)):
                assert_tensor_bytes_equal(full(current.model.layers[1].mlp.router.expert_bias), bias)
                for name in unused:
                    p = current.get_parameter(name)
                    assert p.requires_grad and p.grad is None and p not in current_optimizer.state
            result.update(complete=True, native_continuation_bytes_equal=True,
                          native_next_step_max_error=0., unused_slot_names=unused,
                          saved_slot_names=saved_slots, scheduler=scheduler.state_dict())
    except BaseException as error:
        result["failure"] = repr(error)
        raise
    finally:
        result.update(errors=evidence.errors, steps=evidence.steps, scales=evidence.scales,
                      moments=evidence.moments,
                      routes=evidence.routes, kernel_events=events)
        (directory / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.parametrize("case", LANGUAGE_CASES[:2], ids=["hybrid-q-lora-head-gate", "vl-direct-element-gate"])
def test_native_and_hf_checkpoint_continuation(tmp_path, case):
    write_bailing_source(tmp_path / "source", case)
    mp.spawn(checkpoint_worker, args=(str(tmp_path), case), nprocs=case.world, join=True)
    assert all(json.loads((tmp_path / f"rank-{rank}.json").read_text())["complete"]
               for rank in range(case.world))
