"""Complete built-in Qwen ETP acceptance using an explicit CPU bridge."""

import json
import copy
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.models.qwen3_moe.etp_helpers import (
    Case, Evidence, batch, cpu_qwen_bridge, expected_local, full, install_observers,
    prepare_qwen, reference_model, train_step, write_qwen_source,
)
from tests.engine.backends.fsdp2.etp_checkpoint_helpers import (
    assert_exact_model_and_optimizer, read_hf,
)


def reference_hf(model, case):
    values = {}
    for name, parameter in model.named_parameters():
        prefix = "model.layers.0.mlp.experts"
        if name == prefix + ".weight1":
            for expert, weights in enumerate(parameter.view(case.experts, 16, 2, 12)):
                for index, projection in enumerate(("gate_proj", "up_proj")):
                    values[f"{prefix}.{expert}.{projection}.weight"] = weights[:, index].t().detach()
        elif name == prefix + ".weight2":
            for expert, weight in enumerate(parameter):
                values[f"{prefix}.{expert}.down_proj.weight"] = weight.t().detach()
        else:
            values[name.replace(".mlp.router.", ".mlp.gate.")] = parameter.detach()
    return values


def checkpoint_continuation(directory, engine, model, optimizer, scheduler,
                            reference, ref_optimizer, ref_scheduler, case, evidence):
    checkpoint = directory / "checkpoint"
    generator = torch.Generator().manual_seed(287)
    caller = dict(completed_steps=2, rng=generator.get_state())
    if dist.get_rank() == 0:
        checkpoint.mkdir()
        torch.save(caller, checkpoint / "caller.pt")
    dist.barrier()
    engine.save_model(model, str(checkpoint), final=True)
    engine.save_optimizer_and_scheduler(model, optimizer, scheduler, str(checkpoint))
    expected_hf = reference_hf(reference, case)
    exported = read_hf(checkpoint / "huggingface")
    assert exported.keys() == expected_hf.keys()
    for name, value in exported.items():
        evidence.close(value, expected_hf[name], "HF " + name)
    # Compare export to the actual complete logical trained weights exactly.
    actual_hf = {}
    context = engine.backend._parallel_context
    first_expert = context.ep_mesh.get_local_rank() * (case.experts // case.ep)
    for name, parameter in model.named_parameters():
        value = full(parameter)
        actual_hf[name] = value
        prefix = "model.layers.0.mlp.experts"
        if name == prefix + ".weight1":
            for expert, weights in enumerate(value.view(-1, 16, 2, 12), first_expert):
                for index, projection in enumerate(("gate_proj", "up_proj")):
                    torch.testing.assert_close(exported[f"{prefix}.{expert}.{projection}.weight"],
                                               weights[:, index].t(), rtol=0, atol=0)
        elif name == prefix + ".weight2":
            for expert, weight in enumerate(value, first_expert):
                torch.testing.assert_close(exported[f"{prefix}.{expert}.down_proj.weight"], weight.t(), rtol=0, atol=0)
        else:
            torch.testing.assert_close(exported[name.replace(".mlp.router.", ".mlp.gate.")], value, rtol=0, atol=0)
    restored_engine, restored, restored_optimizer, restored_scheduler = prepare_qwen(directory / "source", case)
    identities = [id(parameter) for parameter in restored.parameters()]
    restored_engine.load_from_checkpoint(restored, str(checkpoint))
    restored_engine.load_optimizer_and_scheduler(restored, restored_optimizer, restored_scheduler, str(checkpoint))
    assert identities == [id(parameter) for parameter in restored.parameters()]
    assert_exact_model_and_optimizer(model, optimizer, restored, restored_optimizer)
    assert scheduler.state_dict() == restored_scheduler.state_dict()
    if case.tied:
        assert restored.lm_head.weight is restored.model.embed_tokens.weight
    restored_reference, restored_ref_optimizer, restored_ref_scheduler = reference_model(directory / "source", case)
    restored_reference.load_state_dict(reference.state_dict())
    restored_ref_optimizer.load_state_dict(copy.deepcopy(ref_optimizer.state_dict()))
    restored_ref_scheduler.load_state_dict(ref_scheduler.state_dict())
    install_observers(restored)
    loaded = torch.load(checkpoint / "caller.pt", weights_only=True)
    assert loaded["completed_steps"] == caller["completed_steps"]
    resumed_generator = torch.Generator()
    resumed_generator.set_state(loaded["rng"])
    next_step = 2 + int(torch.randint(1, 9, (), generator=generator))
    assert next_step == 2 + int(torch.randint(1, 9, (), generator=resumed_generator))
    for objects in ((engine, model, optimizer, scheduler, reference, ref_optimizer, ref_scheduler),
                    (restored_engine, restored, restored_optimizer, restored_scheduler,
                     restored_reference, restored_ref_optimizer, restored_ref_scheduler)):
        train_step(*objects, case, next_step, 1, evidence)
    assert_exact_model_and_optimizer(model, optimizer, restored, restored_optimizer)
    assert scheduler.state_dict() == restored_scheduler.state_dict()
    return dict(completed_steps=3, next_input_step=next_step, native_bytes_equal=True,
                hf_keys=len(exported), saved_local_shapes={name: list(value.shape) for name, value in actual_hf.items()})


def training_worker(rank, directory, case, resume=False):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous",
        rank=rank, world_size=case.world, timeout=timedelta(seconds=120))
    try:
        with cpu_qwen_bridge():
            engine, model, optimizer, scheduler = prepare_qwen(directory / "source", case)
            reference, ref_optimizer, ref_scheduler = reference_model(directory / "source", case)
            context = engine.backend._parallel_context
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(full(parameter), expected_local(reference.get_parameter(name), name, context),
                                           rtol=0, atol=0)
            install_observers(model)
            empty = batch(context.data_dp_mesh.get_local_rank(), 0, 0)
            empty["labels"].fill_(-100)
            with pytest.raises(ValueError, match="no valid targets"):
                engine.prepare_loss_step(model, [empty], torch.device("cpu"))
            assert all(parameter.grad is None for parameter in model.parameters())
            evidence, norms = Evidence(), []
            for step, micros in enumerate((2, 1)):
                norms.append(train_step(engine, model, optimizer, scheduler, reference,
                    ref_optimizer, ref_scheduler, case, step, micros, evidence))
            continuation = None
            if resume:
                continuation = checkpoint_continuation(directory, engine, model, optimizer, scheduler,
                    reference, ref_optimizer, ref_scheduler, case, evidence)
            (directory / f"rank{rank}.json").write_text(json.dumps(dict(rank=rank,
                case=asdict(case), errors=evidence.errors, scales=evidence.scales, norms=norms,
                continuation=continuation, routes=evidence.routes, empty_owners=evidence.empty_owners,
                local_shapes={name: list(parameter.to_local().shape) for name, parameter in model.named_parameters()})))
    finally:
        dist.destroy_process_group()


def run_training(tmp_path, case, *, resume=False):
    write_qwen_source(tmp_path / "source", case)
    mp.spawn(training_worker, args=(str(tmp_path), case, resume), nprocs=case.world, join=True)
    records = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(case.world)]
    assert [record["rank"] for record in records] == list(range(case.world))


@pytest.mark.parametrize("case", [Case(), Case(ep=2), Case(world=8, sp=2),
    Case(world=8, ep=2), Case(world=8, sp=2, ep=2), Case(world=8, ep=4),
    Case(world=8, sp=2, ep=2, experts=2)],
    ids=["E1-Q2-K2", "E2-Q1-K2", "E-lt-SP", "EP-ETP", "SP-EP-ETP", "Q1-K4", "N-lt-Q"])
def test_qwen_aligned_etp_complete_decoder_training(tmp_path, case):
    run_training(tmp_path, case)


def test_qwen_empty_routed_owner_continues_two_steps(tmp_path):
    run_training(tmp_path, Case(world=8, sp=2, ep=2, empty_owner=True))
    records = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(8)]
    owners = [entry for record in records for entry in record["empty_owners"]]
    assert len(owners) == 16
    for owner in owners:
        counts = [owner[field] for field in ("gradient_nonzero", "exp_avg_nonzero", "exp_avg_sq_nonzero")]
        assert all(value == 0 if owner["owner"] == 0 else value > 0 for value in counts)


@pytest.mark.parametrize("sp", [1, 2])
@pytest.mark.parametrize("coefficients", [(0., 0.), (.05, 0.), (0., .03), (.05, .03)],
                         ids=["main", "aux", "z", "aux-z"])
@pytest.mark.parametrize("ac", ["off", "nonreentrant"])
@pytest.mark.parametrize("clip", [False, True], ids=["unclipped", "clipped"])
def test_qwen_etp_objective_ac_clip_and_adamw(tmp_path, sp, coefficients, ac, clip):
    run_training(tmp_path, Case(world=8, sp=sp, ep=2, aux=coefficients[0], z=coefficients[1],
                               ac=ac, clip=clip))


@pytest.mark.parametrize("sp", [1, 2])
def test_qwen_chunked_loss_and_reentrant_control(tmp_path, sp):
    run_training(tmp_path, Case(world=8, sp=sp, ep=2, aux=.05, z=.03,
                               ac="reentrant", clip=True, chunk=True))


@pytest.mark.parametrize("case", [Case(world=8, sp=2, ep=2, aux=.05, z=.03, ac="nonreentrant", clip=True, tied=True),
    Case(world=8, ep=4), Case(world=8, sp=2, ep=2, experts=2)], ids=["Q2-tied", "Q1", "empty-w2"])
def test_qwen_native_resume_hf_and_real_cli(tmp_path, case):
    run_training(tmp_path, case, resume=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path.cwd() / "src") + os.pathsep + str(Path.cwd())
    command = [sys.executable, "-m", "tests.models.qwen3_moe.etp_cli", str(case.world),
        str(tmp_path / "cli-rendezvous"), "--checkpoint_dir", str(tmp_path / "checkpoint"),
        "--output_dir", str(tmp_path / "cli-hf"), "--tt_enable_chunk_loss", "false"]
    completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=150)
    (tmp_path / "cli.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert all(json.loads((tmp_path / f"cli-rank{rank}.json").read_text())["completed"]
               for rank in range(case.world))
    direct, exported = read_hf(tmp_path / "checkpoint" / "huggingface"), read_hf(tmp_path / "cli-hf" / "huggingface")
    assert direct.keys() == exported.keys()
    for name in direct:
        torch.testing.assert_close(direct[name], exported[name], rtol=0, atol=0)
