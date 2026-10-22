"""Complete registered mixed Qwen3.5 decoder acceptance on CPU/Gloo."""

import json
import copy
import os
import subprocess
import sys
from contextlib import ExitStack
from collections import Counter
from unittest.mock import patch
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor, Partial, Replicate

from .etp_helpers import (
    Case, backbone, batch, cpu_bridge, expected_local, full, prepare,
    reference_model, train_step, write_source,
)
from tests.engine.backends.fsdp2.etp_checkpoint_helpers import assert_exact_model_and_optimizer, read_hf
from .precision_helpers import (
    CompleteEvidence, NumericalQualificationError, PrecisionEvidence,
    finish_numerical_qualification, numerical_failure_family,
)


def hf_values(model, case):
    values = {}
    hidden = backbone(model, case).config.hidden_size
    intermediate = backbone(model, case).config.moe_intermediate_size
    for name, parameter in model.named_parameters():
        value = full(parameter)
        if name.endswith(".experts.weight1"):
            value = value.view(-1, hidden, 2, intermediate).permute(0, 2, 3, 1).reshape(-1, 2 * intermediate, hidden)
            name = name.removesuffix("weight1") + "gate_up_proj"
        elif name.endswith(".experts.weight2"):
            value = value.view(-1, intermediate, hidden).transpose(1, 2)
            name = name.removesuffix("weight2") + "down_proj"
        else:
            name = name.replace(".mlp.router.", ".mlp.gate.")
        values[name] = value
    return values


def continuation(directory, objects, case, evidence):
    from torchturbo.engine.checkpoint.metadata import load_metadata

    engine, model, optimizer, scheduler, reference, ref_optimizer, ref_scheduler = objects
    checkpoint = directory / "checkpoint"
    generator = torch.Generator().manual_seed(287)
    caller = dict(completed_steps=2, rng=generator.get_state())
    if dist.get_rank() == 0:
        checkpoint.mkdir()
        torch.save(caller, checkpoint / "caller.pt")
    dist.barrier()
    engine.save_model(model, str(checkpoint), final=True)
    engine.save_optimizer_and_scheduler(model, optimizer, scheduler, str(checkpoint))
    exported, expected = read_hf(checkpoint / "huggingface"), hf_values(reference, case)
    assert exported.keys() == expected.keys()
    source_values = read_hf(directory / "source")
    assert exported.keys() == source_values.keys()
    for name in exported:
        assert exported[name].shape == source_values[name].shape
        evidence.close(exported[name], expected[name], "HF " + name)
    context = engine.backend._parallel_context
    first_expert = context.ep_mesh.get_local_rank() * (case.experts // case.ep)
    for name, value in hf_values(model, case).items():
        actual = exported[name]
        if ".experts." in name:
            actual = actual[first_expert:first_expert + case.experts // case.ep]
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    metadata = load_metadata(str(checkpoint))
    assert metadata["schema_version"] == 2
    layout = metadata["parallel_layout"]
    assert layout["topology"] == dict(world_size=case.world, sp_size=case.sp,
        tp_size=case.tp, ep_size=case.ep, etp_size=case.tp)
    entries = {entry["fqn"]: entry for entry in layout["parameters"]}
    assert entries.keys() == dict(model.named_parameters()).keys()
    for name, parameter in model.named_parameters():
        assert entries[name]["shape"] == list(parameter.shape)
        current_mesh = parameter.device_mesh.mesh.tolist()
        matching = [lane for lane in entries[name]["layouts"] if lane["mesh"] == current_mesh]
        assert len(matching) == 1, name
        assert matching[0]["placements"] == [str(placement) for placement in parameter.placements]
        if ".experts." in name:
            for lane in entries[name]["layouts"]:
                start = lane["ep_rank"] * (case.experts // case.ep)
                assert lane["expert_ids"] == list(range(start, start + case.experts // case.ep))
    restored_objects = prepare(directory / "source", case)
    restored_engine, restored, restored_optimizer, restored_scheduler = restored_objects
    identities = [id(parameter) for parameter in restored.parameters()]
    restored_engine.load_from_checkpoint(restored, str(checkpoint))
    restored_engine.load_optimizer_and_scheduler(restored, restored_optimizer, restored_scheduler, str(checkpoint))
    assert identities == [id(parameter) for parameter in restored.parameters()]
    assert_exact_model_and_optimizer(model, optimizer, restored, restored_optimizer)
    assert scheduler.state_dict() == restored_scheduler.state_dict()
    imported_engine, imported, _, _ = prepare(checkpoint / "huggingface", case)
    for (name, parameter), (_, imported_parameter) in zip(model.named_parameters(), imported.named_parameters()):
        torch.testing.assert_close(full(parameter), full(imported_parameter), rtol=0, atol=0, msg=name)
    restored_reference, restored_ref_optimizer, restored_ref_scheduler = reference_model(directory / "source", case)
    restored_reference.load_state_dict(reference.state_dict())
    restored_ref_optimizer.load_state_dict(copy.deepcopy(ref_optimizer.state_dict()))
    restored_ref_scheduler.load_state_dict(ref_scheduler.state_dict())
    loaded = torch.load(checkpoint / "caller.pt", weights_only=True)
    assert loaded["completed_steps"] == caller["completed_steps"]
    resumed_generator = torch.Generator()
    resumed_generator.set_state(loaded["rng"])
    next_step = 2 + int(torch.randint(1, 9, (), generator=generator))
    assert next_step == 2 + int(torch.randint(1, 9, (), generator=resumed_generator))
    for branch in (objects, (*restored_objects, restored_reference, restored_ref_optimizer, restored_ref_scheduler)):
        train_step(*branch, case, next_step, 1, evidence)
    assert_exact_model_and_optimizer(model, optimizer, restored, restored_optimizer)
    assert scheduler.state_dict() == restored_scheduler.state_dict()
    return dict(hf_keys=len(exported), native_bytes_equal=True, next_input_step=next_step,
                completed_steps=3, layouts=entries)


def training_worker(rank, directory, case, resume=False):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous",
        rank=rank, world_size=case.world, timeout=timedelta(seconds=120))
    try:
        events = []
        with cpu_bridge(events):
            engine, model, optimizer, scheduler = prepare(directory / "source", case)
            reference, ref_optimizer, ref_scheduler = reference_model(directory / "source", case)
            context = engine.backend._parallel_context
            frozen_before = {name: full(parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad}
            if case.wrapper:
                def forbidden_vision(*args, **kwargs):
                    raise AssertionError("Text-only wrapper entered the frozen visual tower")
                model.model.visual.register_forward_pre_hook(forbidden_vision)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(full(parameter), expected_local(reference.get_parameter(name), name, context), rtol=0, atol=0)
            empty = batch(context.data_dp_mesh.get_local_rank(), 0, 0)
            empty["labels"].fill_(-100)
            if case.tp > 1:
                with pytest.raises(ValueError, match="no valid targets"):
                    engine.prepare_loss_step(model, [empty], torch.device("cpu"))
            assert all(parameter.grad is None for parameter in model.parameters())
            evidence, norms = PrecisionEvidence() if case.bf16 else CompleteEvidence(), []
            collective_events = []
            native_redistribute, native_reduce = DTensor.redistribute, dist.reduce_scatter_tensor
            def redistribute(tensor, *args, **kwargs):
                result = native_redistribute(tensor, *args, **kwargs)
                if Partial() in tensor.placements and Replicate() in result.placements:
                    collective_events.append(dict(kind="partial SUM", shape=list(tensor.shape),
                        mesh=tensor.device_mesh.mesh.tolist(), source=str(tensor.dtype), output=str(result.dtype)))
                return result
            def reduce_scatter(*args, **kwargs):
                source = kwargs["input"] if "input" in kwargs else args[1]
                output = kwargs["output"] if "output" in kwargs else args[0]
                collective_events.append(dict(kind="FSDP reduce_scatter", source=str(source.dtype), output=str(output.dtype)))
                return native_reduce(*args, **kwargs)
            with ExitStack() as stack:
                if case.bf16:
                    stack.enter_context(patch.object(DTensor, "redistribute", redistribute))
                    stack.enter_context(patch.object(dist, "reduce_scatter_tensor", reduce_scatter))
                for step, micros in enumerate((2, 1)):
                    norms.append(train_step(engine, model, optimizer, scheduler, reference,
                        ref_optimizer, ref_scheduler, case, step, micros, evidence))
            continued = continuation(directory, (engine, model, optimizer, scheduler, reference,
                ref_optimizer, ref_scheduler), case, evidence) if resume else None
            for name, before in frozen_before.items():
                parameter = model.get_parameter(name)
                assert parameter.grad is None and parameter not in optimizer.state
                torch.testing.assert_close(full(parameter), before, rtol=0, atol=0)
            for name, samples in evidence.expert_magnitudes.items():
                # Routed-output backward contributions exclude padding-only routes.
                layer = int(name.split(".layers.")[1].split(".")[0])
                objective_active = torch.zeros(case.experts, dtype=torch.int64)
                for route in evidence.routes:
                    if route["layer"] == layer:
                        objective_active += torch.tensor(route["expert_objective_active"], dtype=torch.int64)
                dist.all_reduce(objective_active, group=context.dense_fsdp_mesh.get_group())
                local_active = objective_active.chunk(case.ep)[context.ep_mesh.get_local_rank()]
                for magnitude, active in zip((max(column) for column in zip(*samples)), local_active):
                    if active:
                        assert magnitude > 1e-8, (name, samples)
                    else:
                        assert magnitude == 0, (name, samples)
            assert events
            for conv, gdn in zip(events[::2], events[1::2]):
                assert conv["op"] == "causal_conv1d" and gdn["op"] == "gated_delta_rule"
                assert conv["metadata"] is gdn["metadata"]
                assert conv["boundaries"] == gdn["boundaries"]
                bounds = conv["boundaries"]
                assert bounds[:-1] in ([0, 1, 3, 6], [0, 2, 6, 8])
                assert bounds[-1] > bounds[-2]
            serialized_events = [{name: value for name, value in event.items() if name != "metadata"} for event in events]
            if case.bf16:
                assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
                assert all(event["source"] == event["output"] for event in collective_events)
                assert any(event["kind"] == "partial SUM" and event["source"] == "torch.bfloat16" for event in collective_events)
                assert all(event["source"] == "torch.float32" for event in collective_events if event["kind"] == "FSDP reduce_scatter")
                for event in events:
                    if event["op"] == "gated_delta_rule":
                        assert event["dtypes"][:3] == ["torch.bfloat16"] * 3
                        assert event["dtypes"][3] == "torch.float32"
            (directory / f"rank{rank}.json").write_text(json.dumps(dict(rank=rank,
                case=asdict(case), errors=evidence.errors, scales=evidence.scales,
                magnitudes=evidence.magnitudes, relative_errors=evidence.relative_errors,
                expert_magnitudes=evidence.expert_magnitudes,
                failures=evidence.failures if case.bf16 else [], collective_events=collective_events,
                failure_families=dict(Counter(numerical_failure_family(failure[0]) for failure in evidence.failures)) if case.bf16 else {},
                norms=norms, routes=evidence.routes, empty_owners=evidence.empty_owners, continuation=continued, kernel_events=serialized_events,
                target_counts=evidence.target_counts,
                parameter_layouts={name: dict(shape=list(parameter.shape), stride=list(parameter.stride()),
                    local_stride=list(parameter.to_local().stride()), mesh=parameter.device_mesh.mesh.tolist(),
                    gradient_placements=[str(placement) for placement in parameter.grad.placements] if parameter.requires_grad else [],
                    placements=[str(placement) for placement in parameter.placements])
                    for name, parameter in model.named_parameters()},
                local_shapes={name: list(parameter.to_local().shape) for name, parameter in model.named_parameters()})))
    finally:
        dist.destroy_process_group()


def run_training(tmp_path, case, *, resume=False):
    write_source(tmp_path / "source", case)
    mp.spawn(training_worker, args=(str(tmp_path), case, resume), nprocs=case.world, join=True)
    records = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(case.world)]
    assert [record["rank"] for record in records] == list(range(case.world))
    if resume and case.experts == 2:
        assert any(shape[0] == 0 for record in records for name, shape in record["local_shapes"].items()
                   if name.endswith(".experts.weight2"))
    if case.bf16:
        failures = [failure for record in records for failure in record["failures"]]
        finish_numerical_qualification(failures)


@pytest.mark.parametrize("case", [Case(), Case(world=2),
    Case(ep=2, aux=.05, ac="nonreentrant"),
    Case(world=8, sp=2, ep=2, z=.03),
    Case(world=8, sp=2, ep=2, aux=.05, z=.03, ac="nonreentrant", clip=True),
    Case(world=8, sp=2, ep=2, aux=.05, z=.03, ac="reentrant", clip=True, chunk=True),
    Case(world=8, sp=2, ep=2, aux=.05, z=.03, ac="nonreentrant", key_repeat=True),
    Case(world=8, sp=2, ep=2, empty_owner=True),
    Case(world=4, ep=2, wrapper=True),
    Case(world=1, tp=1), Case(world=4, sp=2, ep=2, tp=1)],
    # Training rows perform two updates; persistence has separate cases below.
    ids=["hybrid-no-ep", "hybrid-fsdp-one", "hybrid-ep-aux", "hybrid-sp-z",
         "hybrid-sp-aux-z-clip", "hybrid-sp-chunk-reentrant", "hybrid-sp-key-repeat",
         "hybrid-empty-route-owner", "hybrid-frozen-vision-wrapper", "hybrid-tp1", "hybrid-tp1-sp-ep"])
def test_registered_mixed_decoder_training(tmp_path, case):
    run_training(tmp_path, case)


@pytest.mark.parametrize("experts", [4, 2], ids=["mixed-resume", "empty-storage-resume"])
def test_native_resume_hf_keys_and_next_step(tmp_path, experts):
    case = Case(world=8, sp=2, ep=2, aux=.05, z=.03, ac="nonreentrant", clip=True, experts=experts)
    run_training(tmp_path, case, resume=True)
    if experts == 4:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path.cwd() / "src") + os.pathsep + str(Path.cwd())
        command = [sys.executable, "-m", "tests.models.qwen3_5_moe.etp_cli", str(case.world),
            str(tmp_path / "cli-rendezvous"), "--checkpoint_dir", str(tmp_path / "checkpoint"),
            "--output_dir", str(tmp_path / "cli-hf"), "--tt_enable_chunk_loss", "false"]
        completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=150)
        (tmp_path / "cli.log").write_text(completed.stdout + completed.stderr)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert all(json.loads((tmp_path / f"cli-rank{rank}.json").read_text())["completed"] for rank in range(case.world))
        direct, exported = read_hf(tmp_path / "checkpoint" / "huggingface"), read_hf(tmp_path / "cli-hf" / "huggingface")
        assert direct.keys() == exported.keys()
        for name in direct:
            torch.testing.assert_close(direct[name], exported[name], rtol=0, atol=0)


@pytest.mark.xfail(strict=True, raises=NumericalQualificationError,
    reason="BF16 rowwise QKV rounding amplifies GDN cancellation; dense numerical equivalence is unqualified")
def test_hybrid_sp_bf16_compute_fp32_master(tmp_path):
    run_training(tmp_path, Case(world=8, sp=2, ep=2, ac="nonreentrant", clip=True, bf16=True))
