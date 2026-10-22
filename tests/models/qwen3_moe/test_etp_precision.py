"""Bounded BF16 Qwen ETP semantics on CPU/Gloo, with FP32 master state."""

import json
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor, Partial, Replicate

from tests.models.qwen3_moe.etp_helpers import (
    Case, Evidence, cpu_qwen_bridge, full, install_observers, prepare_qwen,
    reference_model, train_step, write_qwen_source,
)


class PrecisionEvidence(Evidence):
    def __init__(self):
        super().__init__()
        self.magnitudes = {}
        self.relative_errors = {}
        self.relative_error_samples = {}
        self.failures = []

    def close(self, actual, expected, name, **kwargs):
        actual, expected = full(actual).double(), full(expected).double()
        self._close_tensor(actual, expected, name)
        if not name.startswith(("gradient ", "clipped ", "update ", "optimizer ")) or name.endswith(" step"):
            return
        if "experts.weight1" in name:
            actual, expected = actual.reshape(-1, 16, 2, 12), expected.reshape(-1, 16, 2, 12)
        elif "experts.weight2" not in name:
            return
        for index, (left, right) in enumerate(zip(actual, expected)):
            self._close_tensor(left, right, f"{name}/local_expert{index}")

    def _close_tensor(self, actual_tensor, expected_tensor, name):
        magnitude = float(expected_tensor.norm())
        difference = actual_tensor - expected_tensor
        error = float(difference.abs().max()) if difference.numel() else 0.
        self.errors[name] = max(self.errors.get(name, 0.), error)
        self.magnitudes.setdefault(name, []).append(magnitude)
        relative = float(difference.norm()) / magnitude if magnitude else 0.
        self.relative_errors[name] = max(self.relative_errors.get(name, 0.), relative)
        self.relative_error_samples.setdefault(name, []).append(relative)
        # Partitioned BF16 arithmetic and CPU SDPA round at different points.
        # A block-vector error gate is well-defined even at cancelling elements.
        vector_gradient = name.startswith(("gradient", "clipped", "K scaled"))
        if vector_gradient:
            bound = .05 if "probability" in name else .03
            if magnitude == 0:
                if error != 0:
                    self.failures.append(f"Nonzero empty objective gradient: {name}: {error}")
            elif relative > bound:
                self.failures.append(f"Vector gradient error: {name}: {relative} > {bound}")
            if name.startswith(("gradient ", "clipped ")):
                if magnitude == 0:
                    self.failures.append(f"Unexercised reference gradient: {name}")
                else:
                    scale = float((actual_tensor * expected_tensor).sum() / expected_tensor.square().sum())
                    self.scales.setdefault(name.removeprefix("gradient "), []).append(scale)
                    if not .95 < scale < 1.05:
                        self.failures.append(f"Gradient scale: {name} {scale}")
            return
        if name.startswith("optimizer"):
            rtol, atol = (0., 0.) if name.endswith(" step") else (.08, .0002)
        elif name.startswith("update"):
            rtol, atol = .04, .00001
        elif name.startswith("parameter"):
            rtol, atol = .04, .0001
        elif name == "main report":
            rtol, atol = .01, .03
        elif name.startswith(("gradient", "clipped")) or name == "norm":
            rtol, atol = .06, .0007
        elif "probability gradient" in name:
            rtol, atol = .03, .0001
        elif name == "K scaled input":
            rtol, atol = .06, .0028
        else:
            rtol, atol = .03, .03
        if ".self_attn.q_norm." in name or ".self_attn.k_norm." in name:
            if name.startswith(("gradient", "clipped")):
                rtol, atol = .03, .00002
            elif name.startswith("update"):
                rtol, atol = .03, .0000005
        try:
            super().close(actual_tensor, expected_tensor, name, rtol=rtol, atol=atol)
        except AssertionError as error:
            self.failures.append(str(error))
        if name.startswith("update"):
            if magnitude == 0 or relative > .04:
                self.failures.append(f"Update non-vacuity/error: {name} magnitude={magnitude} error={relative}")
            if "_norm.weight" in name and magnitude < 1e-6:
                self.failures.append(f"Head norm update is too small: {name} {magnitude}")
        if name.startswith("optimizer") and not name.endswith(" step") and (magnitude == 0 or relative > .04):
            self.failures.append(f"Optimizer moment non-vacuity/error: {name} magnitude={magnitude} error={relative}")

    def gradient(self, actual, expected, name):
        self.close(actual, expected, "gradient " + name)


def precision_worker(rank, directory):
    torch.set_num_threads(1)
    case = Case(world=8, sp=2, ep=2, ac="nonreentrant", clip=True, bf16=True)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous",
        rank=rank, world_size=8, timeout=timedelta(seconds=120))
    try:
        with cpu_qwen_bridge():
            source = Path(directory) / "source"
            engine, model, optimizer, scheduler = prepare_qwen(source, case)
            master, master_optimizer, master_scheduler = reference_model(source, case)
            install_observers(model)
            context = engine.backend._parallel_context
            events, norms, evidence = [], [], PrecisionEvidence()
            native_redistribute, native_reduce = DTensor.redistribute, dist.reduce_scatter_tensor

            def redistribute(tensor, *args, **kwargs):
                desired = kwargs.get("placements", args[1] if len(args) > 1 else None)
                result = native_redistribute(tensor, *args, **kwargs)
                if tensor.placements == (Partial(),) and desired is not None and tuple(desired) == (Replicate(),):
                    if tensor.shape == (4,) and tensor.device_mesh == context.tp_mesh:
                        kind = "head norm TP SUM"
                    elif tensor.ndim == 2 and tensor.device_mesh == context.etp_mesh:
                        kind = "expert output ETP SUM"
                    else:
                        return result
                    events.append((kind, str(tensor.dtype), str(result.dtype)))
                    assert tensor.dtype == result.dtype == torch.bfloat16
                return result

            def reduce_scatter(*args, **kwargs):
                source = kwargs["input"] if "input" in kwargs else args[1]
                output = kwargs["output"] if "output" in kwargs else args[0]
                group = kwargs.get("group", args[3] if len(args) > 3 else None)
                if group is context.dense_fsdp_mesh.get_group():
                    kind = "dense FP32 reduction"
                else:
                    assert group is context.expert_fsdp_mesh.get_group()
                    kind = "expert FP32 reduction"
                events.append((kind, str(source.dtype), str(output.dtype)))
                assert source.dtype == output.dtype == torch.float32
                return native_reduce(*args, **kwargs)

            with patch.object(DTensor, "redistribute", redistribute), \
                    patch.object(dist, "reduce_scatter_tensor", reduce_scatter):
                for step, micros in enumerate((2, 1)):
                    norms.append(train_step(engine, model, optimizer, scheduler, master,
                        master_optimizer, master_scheduler, case, step, micros, evidence))
            kinds = [event[0] for event in events]
            assert kinds.count("head norm TP SUM") == 6
            assert kinds.count("expert output ETP SUM") >= 3, events
            assert "dense FP32 reduction" in kinds and "expert FP32 reduction" in kinds
            pending_norms = 0
            for kind in kinds:
                if kind == "head norm TP SUM":
                    pending_norms += 1
                elif kind == "dense FP32 reduction" and pending_norms:
                    assert pending_norms == 2
                    pending_norms = 0
            assert pending_norms == 0
            assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
            Path(directory, f"rank{rank}.json").write_text(json.dumps(dict(rank=rank,
                case=asdict(case), errors=evidence.errors, scales=evidence.scales,
                relative_errors=evidence.relative_errors, magnitudes=evidence.magnitudes,
                relative_error_samples=evidence.relative_error_samples,
                expert_owner=context.ep_mesh.get_local_rank(),
                norms=norms, events=events, failures=evidence.failures, routes=evidence.routes)))
            assert not evidence.failures, evidence.failures
    finally:
        dist.destroy_process_group()


def test_qwen_bf16_sp_ac_clip_two_step_semantics(tmp_path):
    case = Case(world=8, sp=2, ep=2, ac="nonreentrant", clip=True, bf16=True)
    write_qwen_source(tmp_path / "source", case)
    mp.spawn(precision_worker, args=(str(tmp_path),), nprocs=8, join=True)
    assert all((tmp_path / f"rank{rank}.json").exists() for rank in range(8))
