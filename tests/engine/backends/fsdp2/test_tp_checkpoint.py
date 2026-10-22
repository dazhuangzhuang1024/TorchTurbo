"""Native dense TP checkpoint contracts on CPU/Gloo."""

import copy
import json
import os
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from transformers import PretrainedConfig

from torchturbo.engine.backends.fsdp2.checkpoint.manager import FSDP2CheckpointManager
from torchturbo.engine.backends.fsdp2.checkpoint.metadata import build_parallel_layout
from torchturbo.engine.checkpoint.coordinator import CheckpointCoordinator
from torchturbo.engine.checkpoint.metadata import load_metadata, save_metadata
from torchturbo.engine.config import EngineConfig
from torchturbo.engine.backends.fsdp2.parallel_context import (
    ParallelContext,
    ParameterOwnership,
)
from torchturbo.runtime.cpu import CpuRuntime


class Decoder(nn.Module):
    """A dense decoder block with composite and FSDP-only parameters."""

    def __init__(self, width=6, device="cpu"):
        super().__init__()
        self.config = PretrainedConfig()
        self.up = nn.Linear(5, width, device=device, dtype=torch.float64)
        self.down = nn.Linear(width, 4, device=device, dtype=torch.float64)
        self.scale = nn.Parameter(torch.ones(4, device=device, dtype=torch.float64))
        self.scale_alias = self.scale

    def forward(self, value):
        return self.down(torch.nn.functional.silu(self.up(value))) * self.scale

    def tie_weights(self):
        self.scale_alias = self.scale


def _prepare(context, width=6, device="cpu"):
    model = Decoder(width, device)
    parallelize_module(model, context.tp_mesh, {
        "up": ColwiseParallel(), "down": RowwiseParallel(input_layouts=Shard(-1)),
    })
    fully_shard(model, mesh=context.dense_fsdp_mesh)
    ownership = ParameterOwnership()
    ownership.register(model, set())
    manager = FSDP2CheckpointManager(
        rank=dist.get_rank(), parallel_context=context, ownership=ownership,
        device_runtime=CpuRuntime(), cpu_offload=False,
        require_optimizer_not_suspended=lambda: None,
    )
    config = EngineConfig.from_dict({"tp_size": 2, "checkpoint": {"hf_export": "final"}})
    coordinator = CheckpointCoordinator(config, manager, parallel_context={})
    return model, manager, coordinator


def _step(model, optimizer, step):
    generator = torch.Generator().manual_seed(123 + step)
    inputs = torch.randn(2, 5, generator=generator, dtype=torch.float64)
    target = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    optimizer.zero_grad()
    (model(inputs) - target).square().mean().backward()
    optimizer.step()


def _assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        a, b = left[name], right[name]
        if isinstance(a, dict):
            _assert_state_equal(a, b)
        elif isinstance(a, torch.Tensor):
            if isinstance(a, DTensor):
                assert a.device_mesh == b.device_mesh
                assert a.placements == b.placements
                a, b = a.to_local(), b.to_local()
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            assert a == b


def _run_parallel_layout(rank, init_file, mode):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=4, timeout=timedelta(seconds=90))
    try:
        context = ParallelContext("cpu", sp_size=1, ep_size=1, tp_size=2)
        model = nn.Module()
        model.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(64)])
        model.scale = nn.Parameter(torch.ones(4))
        model.scale_alias = model.scale
        parallelize_module(model, context.tp_mesh, {
            f"blocks.{index}": ColwiseParallel() for index in range(64)
        })
        fully_shard(model, mesh=context.dense_fsdp_mesh)
        ownership = ParameterOwnership()
        ownership.register(model, set())
        if mode != "buffers":
            if rank == 1:
                if mode == "order":
                    model.blocks._modules = dict(reversed(model.blocks._modules.items()))
                else:
                    del model.blocks[0]
            with pytest.raises(ValueError, match="parameter names.*order and count"):
                build_parallel_layout(model, context, ownership)
            return

        # Observe real native serialized collective buffers without replacing transport.
        buffer_bytes = []
        native_all_gather = c10d.all_gather
        native_broadcast = c10d.broadcast

        def observe_all_gather(outputs, tensor, *args, **kwargs):
            if tensor.dtype == torch.uint8:
                buffer_bytes.append(sum(output.numel() for output in outputs))
            return native_all_gather(outputs, tensor, *args, **kwargs)

        def observe_broadcast(tensor, *args, **kwargs):
            if tensor.dtype == torch.uint8:
                buffer_bytes.append(tensor.numel())
            return native_broadcast(tensor, *args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(c10d, "all_gather", observe_all_gather)
            patch.setattr(c10d, "broadcast", observe_broadcast)
            layout = build_parallel_layout(model, context, ownership)

        expected_parameters = [{
            "fqn": f"blocks.{index}.weight", "aliases": [f"blocks.{index}.weight"],
            "shape": [4, 4], "ownership": "dense", "layouts": [{
                "mesh": [[0, 1], [2, 3]], "mesh_dim_names": ["dp", "tp"],
                "placements": ["_S(0, 2)", "S(0)"],
            }],
        } for index in range(64)]
        expected_parameters.append({
            "fqn": "scale", "aliases": ["scale", "scale_alias"],
            "shape": [4], "ownership": "dense", "layouts": [
                {"mesh": [0, 2], "mesh_dim_names": ["dp"], "placements": ["S(0)"]},
                {"mesh": [1, 3], "mesh_dim_names": ["dp"], "placements": ["S(0)"]},
            ],
        })
        expected = {
            "layout_schema_version": 1, "expert_layout_version": "packed_2d_v1",
            "topology": {"world_size": 4, "sp_size": 1, "tp_size": 2,
                         "ep_size": 1, "etp_size": 1},
            "parameters": sorted(expected_parameters, key=lambda entry: entry["fqn"]),
        }
        assert layout == expected
        # Canonical descriptor ordering is part of the saved metadata format.
        for entry in expected["parameters"]:
            entry["layouts"] = [json.loads(json.dumps(item, sort_keys=True))
                                for item in entry["layouts"]]
        assert json.dumps(layout) == json.dumps(expected)
        Path(init_file).with_name(f"layout-rank-{rank}.json").write_text(json.dumps({
            "collective_buffer_bytes": buffer_bytes, "layout": layout,
        }))
        layout["parameters"][0]["layouts"][0]["mesh"][0][0] = -1
        assert layout["parameters"][1] == expected["parameters"][1]
        assert buffer_bytes
        # Repeated mesh descriptions must not fill every WORLD receive slot.
        assert sum(buffer_bytes) < 10_000, buffer_bytes
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_parallel_layout_bounds_native_collective_buffers_and_preserves_schema(tmp_path):
    mp.spawn(_run_parallel_layout, args=(str(tmp_path / "init"), "buffers"), nprocs=4, join=True)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize("mode", ["order", "count"])
def test_parallel_layout_rejects_parameter_name_mismatch_on_every_rank(tmp_path, mode):
    mp.spawn(_run_parallel_layout, args=(str(tmp_path / "init"), mode), nprocs=4, join=True)


def _run_checkpoint(rank, init_file, directory, width):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=4, timeout=timedelta(seconds=90))
    try:
        context = ParallelContext("cpu", sp_size=1, ep_size=1, tp_size=2)
        assert context.expert_checkpoint_owner_ranks == [0, 1, 2, 3]
        model, manager, coordinator = _prepare(context, width)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, foreach=False)
        for step in range(2):
            _step(model, optimizer, step)
        coordinator.save_model(model, directory, source_path="/models/source")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        scheduler.step()
        coordinator.save_optimizer_and_scheduler(model, optimizer, scheduler, directory)
        optimizer_metadata = FileSystemReader(manager.optimizer_dir(directory)).read_metadata()
        params_key = "optimizer.param_groups.0.params"
        assert params_key in optimizer_metadata.state_dict_metadata
        assert optimizer_metadata.planner_data[params_key] == ("optimizer", "param_groups", 0, "params")
        assert not any(key.startswith(f"{params_key}.") for key in optimizer_metadata.state_dict_metadata)
        metadata = load_metadata(directory)
        assert metadata["schema_version"] == 2
        layout = metadata["parallel_layout"]
        assert layout["topology"] == {"world_size": 4, "sp_size": 1, "tp_size": 2,
                                       "ep_size": 1, "etp_size": 1}
        assert layout["expert_layout_version"] == "packed_2d_v1"
        entries = {entry["fqn"]: entry for entry in layout["parameters"]}
        assert entries["up.weight"]["layouts"][0]["placements"] == ["_S(0, 2)", "S(0)"]
        assert entries["scale"]["aliases"] == ["scale", "scale_alias"]
        assert sorted(item["mesh"] for item in entries["scale"]["layouts"]) == [[0, 2], [1, 3]]
        assert entries["up.weight"]["layouts"][0]["mesh"] == [[0, 1], [2, 3]]
        assert all("::tp" not in key for key in FileSystemReader(manager.native_model_dir(directory)).read_metadata().state_dict_metadata)
        restored, _, restore_coordinator = _prepare(context, width)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.003, foreach=False)
        identities = [id(param) for param in restored.parameters()]
        restore_coordinator.load_model(restored, directory, source_path="/models/source")
        restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.9)
        restore_coordinator.load_optimizer_and_scheduler(restored, restored_optimizer, restored_scheduler, directory)
        assert scheduler.state_dict() == restored_scheduler.state_dict()
        assert [id(param) for param in restored.parameters()] == identities
        assert restored.scale is restored.scale_alias
        _assert_state_equal(model.state_dict(), restored.state_dict())
        _assert_state_equal(optimizer.state_dict(), restored_optimizer.state_dict())
        _step(model, optimizer, 2)
        _step(restored, restored_optimizer, 2)
        _assert_state_equal(model.state_dict(), restored.state_dict())
        _assert_state_equal(optimizer.state_dict(), restored_optimizer.state_dict())
        original_metadata = copy.deepcopy(metadata)
        for mode in ("degree", "config_degree", "placement", "shape", "ownership", "aliases", "version", "schema1", "missing"):
            saved = copy.deepcopy(original_metadata)
            if mode == "degree":
                saved["parallel_layout"]["topology"]["tp_size"] = 1
            elif mode == "config_degree":
                saved["engine_config"]["parallel_plan"]["tp_size"] = 1
            elif mode == "version":
                saved["parallel_layout"]["expert_layout_version"] = "gated_logical_v1"
            elif mode == "schema1":
                saved["schema_version"] = 1
                del saved["parallel_layout"]
            elif mode != "missing":
                entry = saved["parallel_layout"]["parameters"][0]
                if mode == "placement":
                    entry["layouts"][0]["placements"] = ["Replicate()", "Replicate()"]
                elif mode == "shape":
                    entry["shape"] = [entry["shape"][0] + 1]
                elif mode == "ownership":
                    entry["ownership"] = "expert"
                else:
                    entry["aliases"] = ["changed"]
            if rank == 0:
                if mode == "missing":
                    (Path(directory) / "torchturbo" / "checkpoint.json").unlink()
                    (Path(directory) / "torchturbo").rmdir()
                else:
                    save_metadata(saved, directory)
            dist.barrier()
            for component in ("model", "optimizer"):
                fresh_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.003)
                fresh_scheduler = torch.optim.lr_scheduler.StepLR(fresh_optimizer, step_size=1)
                scheduler_before = fresh_scheduler.state_dict()
                before = {name: value.to_local().clone() for name, value in restored.state_dict().items()}
                with pytest.MonkeyPatch.context() as patch:
                    import torch.distributed.checkpoint as dcp
                    patch.setattr(dcp, "load", lambda *args, **kwargs: pytest.fail("DCP load entered before metadata validation"))
                    with pytest.raises((ValueError, RuntimeError), match="layout|topology|HF|metadata"):
                        if component == "model":
                            restore_coordinator.load_model(restored, directory, source_path="/models/source")
                        else:
                            restore_coordinator.load_optimizer_and_scheduler(restored, fresh_optimizer, fresh_scheduler, directory)
                assert fresh_scheduler.state_dict() == scheduler_before
                assert not fresh_optimizer.state
                for name, value in restored.state_dict().items():
                    torch.testing.assert_close(value.to_local(), before[name], rtol=0, atol=0)
        if rank == 0:
            save_metadata(original_metadata, directory)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize("width", [6, 2], ids=["uneven", "empty"])
def test_dense_tp_native_adamw_roundtrip_and_metadata_preflight(tmp_path, width):
    mp.spawn(_run_checkpoint, args=(str(tmp_path / "init"), str(tmp_path / "checkpoint"), width),
             nprocs=4, join=True)


def _run_hf(rank, init_file, directory, width):
    from safetensors.torch import load_file, save_file
    from torchturbo.engine.backends.fsdp2.checkpoint.weight_loader import DistParamLoader
    from torchturbo.models.weight_converter import BaseWeightConverter

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=4, timeout=timedelta(seconds=90))
    try:
        context = ParallelContext("cpu", sp_size=1, ep_size=1, tp_size=2)
        source = Path(directory) / "source"
        expected = {
            name: torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape) / 100
            for name, parameter in Decoder(width).named_parameters()
        }
        expected["scale"] += 1
        if rank == 0:
            source.mkdir()
            save_file(expected, str(source / "model.safetensors"))
            (source / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": sum(value.numel() * 4 for value in expected.values())},
                "weight_map": {name: "model.safetensors" for name in expected},
            }))
            (source / "config.json").write_text('{"model_type": "checkpoint_decoder"}')
        dist.barrier()
        model, manager, _ = _prepare(context, width, "meta")
        model.all_tied_weights_keys = {"scale_alias": "scale"}
        converter = BaseWeightConverter(model, {"ep": context.get_ep_info()}, str(source))
        DistParamLoader._BROADCAST_MEM_BUDGET_CAP = 64
        manager.load_hf(model, str(source), weight_converter=converter, assign=True)
        assert model.scale is model.scale_alias
        for name, parameter in model.named_parameters():
            assert not parameter.is_meta
            torch.testing.assert_close(parameter.full_tensor(), expected[name].to(torch.float64), rtol=0, atol=0)
        assert model.up.weight.device_mesh == context.dense_mesh
        assert model.scale.device_mesh == context.dense_fsdp_mesh
        manager._ownership.bind(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, foreach=False)
        _step(model, optimizer, 0)
        full_values = {name: parameter.full_tensor().float() for name, parameter in model.named_parameters()}
        export = str(Path(directory) / "export")
        manager.save_hf(model, export, source_path=str(source), weight_converter=converter)
        index = json.loads((Path(export) / "model.safetensors.index.json").read_text())
        exported = {}
        for filename in set(index["weight_map"].values()):
            exported.update(load_file(str(Path(export) / filename)))
        assert set(exported) == set(expected)
        for name in exported:
            torch.testing.assert_close(exported[name], full_values[name], rtol=0, atol=0)
        identities = [id(parameter) for parameter in model.parameters()]
        manager.load_hf(model, export, weight_converter=converter, assign=False)
        assert [id(parameter) for parameter in model.parameters()] == identities
        assert model.scale is model.scale_alias
        assert set(optimizer.state) == set(model.parameters())
        _step(model, optimizer, 1)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize("width", [6, 2], ids=["uneven", "empty"])
def test_composite_hf_assignment_export_and_resume(tmp_path, width):
    mp.spawn(_run_hf, args=(str(tmp_path / "init"), str(tmp_path), width), nprocs=4, join=True)
