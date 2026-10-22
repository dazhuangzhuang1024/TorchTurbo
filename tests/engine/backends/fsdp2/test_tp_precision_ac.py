"""Bounded BF16 and reentrant dense TP/FSDP2 checks on CPU/Gloo.

The attention bridge delegates to the registered maximum-logit CPU provider.
It does not establish unmodified production CPU attention or accelerator support.
"""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.distributed.tensor import DTensor, Partial, Replicate

from tests.engine.backends.fsdp2 import test_tp_integration as fixture


# Fixed before the successful preparation probe; do not fit degree factors.
_BF16_TOLERANCES = {
    "logits": (.03, .03),
    "report": (.01, .03),
    "gradient": (.06, .0007),
    "norm": (.06, .0007),
    "parameter": (.04, .0001),
    "update delta": (.04, .0001),
    "moment": (.08, .0002),
    "head norm gradient": (.03, .00002),
    "head norm update": (.03, .000005),
    "step": (0, 0),
}
_HEAD_NORMS = ("layers.0.attn.q_norm.weight", "layers.0.attn.k_norm.weight")


def _assert_bf16_close(actual, expected, category, name, errors):
    actual, expected = fixture._full(actual), fixture._full(expected)
    rtol, atol = _BF16_TOLERANCES[category]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)
    difference = float((actual - expected).abs().max())
    errors[category] = max(errors.get(category, 0.0), difference)


def _bf16_oracle(master, micros, dense_size):
    """Compute rounded BF16 leaf gradients while retaining FP32 masters."""
    batches, count = {}, 0
    for data_rank in range(2):
        for micro in micros:
            batch = fixture._batch(data_rank, micro, 0)
            # Reproduce the declared hook transform without invoking the hook.
            batch["labels"][0, 4] = -100
            targets = torch.full_like(batch["labels"], -100)
            for start, end in ((0, 1), (1, 3), (3, 6)):
                targets[:, start:end - 1] = batch["labels"][:, start + 1:end]
            count += int((targets != -100).sum())
            batches[data_rank, micro] = batch, targets
    assert count == 5
    gradients = {name: torch.zeros_like(parameter)
                 for name, parameter in master.named_parameters()}
    report, logits = 0, {}
    for key, (batch, targets) in batches.items():
        # Separate BF16 leaves round gradients before FP32 dense averaging.
        # Autocast over FP32 leaves would not model FSDP parameter casting.
        compute = fixture.PackedDenseDecoder().to(torch.bfloat16)
        compute.load_state_dict(master.state_dict())
        assert all(parameter.dtype == torch.bfloat16 for parameter in compute.parameters())
        output = compute(**batch).logits
        assert output.dtype == torch.bfloat16
        logits[key] = output.detach().clone()
        # CPU native CE returns BF16. Reduce over the full row once to match
        # its rounding boundary while deriving packed targets independently.
        raw_loss = F.cross_entropy(output.reshape(-1, 24), targets.reshape(-1),
                                   reduction="sum", ignore_index=-100)
        assert raw_loss.dtype == torch.bfloat16
        loss = raw_loss / torch.tensor(count / dense_size, dtype=torch.float32)
        report = report + raw_loss.float().detach() / count
        loss.backward()
        for name, parameter in compute.named_parameters():
            assert parameter.grad.dtype == torch.bfloat16
            # Only the independent reference reproduces native dense averaging.
            gradients[name] += parameter.grad.float() / dense_size
    for name, parameter in master.named_parameters():
        assert parameter.dtype == torch.float32
        parameter.grad = gradients[name]
    return count, report, logits


def _bf16_step(engine, model, optimizer, master, master_optimizer, errors):
    micros = (0, 1, 2)
    context = engine.backend._parallel_context
    data_rank = context.get_dp_info()["dp_rank"]
    initial = {name: parameter.detach().clone()
               for name, parameter in master.named_parameters()}
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(fixture._full(parameter), initial[name], rtol=0, atol=0)
    optimizer.zero_grad(set_to_none=True)
    master_optimizer.zero_grad(set_to_none=True)
    count, expected_report, expected_logits = _bf16_oracle(master, micros, context.dense_fsdp_mesh.size())
    sources = [fixture._batch(data_rank, micro, 0) for micro in micros]
    before_calls = model.prepare_calls
    prepared, actual_count = engine.prepare_loss_step(model, sources, torch.device("cpu"))
    assert actual_count == count
    assert model.prepare_calls == before_calls + len(micros)
    report = 0
    for micro, batch in zip(micros, prepared):
        loss, output = engine.compute_prepared_loss(
            model, batch, True, num_items_in_batch=count,
            gradient_accumulation_steps=len(micros),
        )
        assert output.logits.dtype == torch.bfloat16
        assert not output.main_loss.requires_grad
        report = report + output.main_loss
        _assert_bf16_close(output.logits[:, :6], expected_logits[data_rank, micro][:, :6],
                           "logits", f"valid logits {micro}", errors)
        loss.backward()
    _assert_bf16_close(report, expected_report, "report", "optimizer-step report", errors)
    scales = {}
    for name, parameter in model.named_parameters():
        expected = master.get_parameter(name)
        assert parameter.grad is not None, name
        assert parameter.grad.dtype == torch.float32
        _assert_bf16_close(parameter.grad, expected.grad, "gradient", name, errors)
        actual_gradient = fixture._full(parameter.grad).double()
        expected_gradient = expected.grad.double()
        denominator = expected_gradient.square().sum()
        assert denominator > 0, name
        scale = float((actual_gradient * expected_gradient).sum() / denominator)
        assert .95 < scale < 1.05, (name, scale)
        scales[name] = scale
        if name in _HEAD_NORMS:
            assert expected_gradient.abs().min() > .001
            _assert_bf16_close(parameter.grad, expected.grad, "head norm gradient", name, errors)
    expected_norm = torch.linalg.vector_norm(torch.cat([
        parameter.grad.flatten() for parameter in master.parameters()
    ]))
    actual_norm = engine.backend.clip_grad_norm_(model.parameters(), float("inf"))
    _assert_bf16_close(actual_norm, expected_norm, "norm", "logical gradient norm", errors)
    optimizer.step()
    master_optimizer.step()
    for name, parameter in model.named_parameters():
        expected = master.get_parameter(name)
        assert parameter.dtype == torch.float32
        _assert_bf16_close(parameter, expected, "parameter", name, errors)
        actual_delta = fixture._full(parameter) - initial[name]
        expected_delta = expected.detach() - initial[name]
        _assert_bf16_close(actual_delta, expected_delta, "update delta", name, errors)
        if name in _HEAD_NORMS:
            assert actual_delta.abs().min() > .00004
            _assert_bf16_close(actual_delta, expected_delta, "head norm update", name, errors)
        state, expected_state = optimizer.state[parameter], master_optimizer.state[expected]
        assert state.keys() == expected_state.keys()
        for key in state:
            assert state[key].dtype == torch.float32
            category = "step" if key == "step" else "moment"
            _assert_bf16_close(state[key], expected_state[key], category, f"{name} {key}", errors)
    return scales


def _assert_reduction_events(events, num_micros):
    partial_calls, reduce_calls, pending_norms = 0, 0, 0
    for kind, input_dtype, output_dtype in events:
        if kind == "partial to replicate":
            assert input_dtype == output_dtype == "torch.bfloat16"
            partial_calls += 1
            pending_norms += 1
        else:
            assert kind == "dense reduce scatter"
            assert input_dtype == output_dtype == "torch.float32"
            reduce_calls += 1
            if pending_norms:
                # Both norms are restored before their owning unit's dense
                # reduction, rather than relying on the final master dtype.
                assert pending_norms == 2
                pending_norms = 0
    assert partial_calls == 2 * num_micros
    assert reduce_calls > 0
    assert pending_norms == 0


def _worker(rank, rendezvous, directory, sp_size, bf16):
    from torchturbo.models.modules.attention.ulysses import ulysses
    from torchturbo.models.modules.attention.interface import attention_forward

    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=4 * sp_size, timeout=timedelta(seconds=120))
    calls, events, errors = [], [], {}
    fixture._MAX_ERRORS.clear()

    def cpu_attention_bridge(*values, **options):
        assert not options.pop("return_max_logits", False)
        calls.append((tuple(values[0].shape), str(values[0].dtype), torch.is_grad_enabled()))
        output, _ = attention_forward(*values, return_max_logits=True, **options)
        return output

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ulysses, "attention_forward", cpu_attention_bridge)
            source = Path(directory) / "source"
            if rank == 0:
                fixture._source(source, False)
            dist.barrier()
            # A larger epsilon conditions independent BF16 gradient rounding
            # errors before AdamW's first update. Existing FP32 defaults stay.
            eps = .1 if bf16 else 1e-6
            engine, model, optimizer, scheduler = fixture._prepare(
                source, sp_size, False, not bf16, False,
                mixed_precision="bf16" if bf16 else "fp32",
                use_reentrant=not bf16, optimizer_eps=eps,
            )
            master, master_optimizer, master_scheduler = fixture._reference(
                source, False, optimizer_eps=eps,
            )
            context = engine.backend._parallel_context
            if bf16:
                original_redistribute = DTensor.redistribute
                original_reduce_scatter = dist.reduce_scatter_tensor

                def observe_redistribute(tensor, *args, **kwargs):
                    desired = kwargs.get("placements", args[1] if len(args) > 1 else None)
                    is_head_norm = (
                        tensor.shape == torch.Size([2])
                        and tensor.device_mesh == context.tp_mesh
                        and tensor.placements == (Partial(),)
                        and desired == (Replicate(),)
                    )
                    result = original_redistribute(tensor, *args, **kwargs)
                    if is_head_norm:
                        events.append(("partial to replicate", str(tensor.dtype), str(result.dtype)))
                    return result

                def observe_reduce_scatter(*args, **kwargs):
                    input_tensor = kwargs["input"] if "input" in kwargs else args[1]
                    output_tensor = kwargs["output"] if "output" in kwargs else args[0]
                    group = kwargs.get("group", args[3] if len(args) > 3 else None)
                    assert group is context.dense_fsdp_mesh.get_group()
                    events.append(("dense reduce scatter", str(input_tensor.dtype), str(output_tensor.dtype)))
                    return original_reduce_scatter(*args, **kwargs)

                # These public observers forward unchanged native operations;
                # no private FSDP state or native gradients are modified.
                patch.setattr(DTensor, "redistribute", observe_redistribute)
                patch.setattr(dist, "reduce_scatter_tensor", observe_reduce_scatter)
                scales = _bf16_step(engine, model, optimizer, master, master_optimizer, errors)
                _assert_reduction_events(events, 3)
                assert len(calls) == 3
                assert all(dtype == "torch.bfloat16" and enabled for _, dtype, enabled in calls)
            else:
                scales = {}
                for step, micros in enumerate(((0, 1, 2), (2,))):
                    _, count = fixture._train_step(
                        engine, model, optimizer, scheduler, master,
                        master_optimizer, master_scheduler, step, micros, False,
                    )
                    assert count == (5 if step == 0 else 3)
                errors = dict(fixture._MAX_ERRORS)
                assert len(calls) == 8
                assert [enabled for _, _, enabled in calls] == [False, True] * 4
                assert all(dtype == "torch.float32" for _, dtype, _ in calls)
            assert all(shape[2] == 8 // (2 * sp_size) for shape, _, _ in calls)
            evidence = {"errors": errors, "scales": scales, "events": events, "attention_calls": calls}
            rank_evidence = [None] * dist.get_world_size()
            dist.all_gather_object(rank_evidence, evidence)
            if rank == 0:
                mode = "bf16" if bf16 else "reentrant"
                Path(directory, "evidence.json").write_text(json.dumps(rank_evidence, indent=2))
                maxima = {category: max(item["errors"][category] for item in rank_evidence)
                          for category in errors}
                scale_values = [scale for item in rank_evidence for scale in item["scales"].values()]
                print(f"DENSE PRECISION AC {mode} S{sp_size}: {json.dumps(maxima, sort_keys=True)}", flush=True)
                if scale_values:
                    print(f"GRADIENT SCALE RANGE: {min(scale_values):.9f}..{max(scale_values):.9f}", flush=True)
                print(f"NATIVE DTYPE EVENTS: {json.dumps(events)}", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_dense_tp_bf16_gradients_and_adamw(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), 1, True),
             nprocs=4, join=True)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("sp_size", [1, 2], ids=["D2-S1-T2", "D2-S2-T2"])
def test_dense_tp_reentrant_checkpoint_training(tmp_path, sp_size):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), sp_size, False),
             nprocs=4 * sp_size, join=True)
