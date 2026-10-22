"""Real logical expert HF/native checkpoints, ownership and resume preflight."""

import copy
import json
import pickle
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.tensor import Shard

from torchturbo.engine.checkpoint.metadata import load_metadata, save_metadata
from torchturbo.engine.checkpoint import CheckpointFormat
from torchturbo.ops import get_or_create_ops
from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext
from torchturbo.runtime.cpu import CpuRuntime
from tests.engine.backends.fsdp2.etp_checkpoint_helpers import (
    CheckpointMoE, assert_exact_model_and_optimizer, expert_values, hf_values,
    prepare_checkpoint_model, prepare_checkpoint_optimizer, read_hf, training_step, write_source,
)


def _initialize(rank, directory, world, sp=1):
    torch.set_num_threads(1)
    torch.manual_seed(715 + rank // 2)
    dist.init_process_group("gloo", init_method=f"file://{directory}/store", rank=rank,
                            world_size=world, timeout=timedelta(seconds=180))
    get_or_create_ops(device_runtime=CpuRuntime())
    return ParallelContext("cpu", sp_size=sp, ep_size=2, tp_size=2, etp_size=2)


def _layout_evidence(model, context, num_local_experts):
    evidence = {}
    q = context.expert_fsdp_mesh.size()
    for index, parameter in ((1, model.moe.experts.weight1), (2, model.moe.experts.weight2)):
        assert parameter.placements == (Shard(0), Shard(2 if index == 1 else 1))
        assert parameter.device_mesh == context.expert_mesh["edp", "etp"]
        expected_shape = (num_local_experts * 3, 2, 4) if index == 1 else (num_local_experts, 4, 3)
        assert tuple(parameter.shape) == expected_shape
        assert tuple(parameter.stride()) == ((8, 4, 1) if index == 1 else (12, 3, 1))
        rows = num_local_experts * 3 if index == 1 else num_local_experts
        chunk = (rows + q - 1) // q
        local_rows = max(0, min(chunk, rows - context.expert_fsdp_mesh.get_local_rank() * chunk))
        assert tuple(parameter.to_local().shape) == (local_rows, 2, 2 if index == 1 else 3)
        evidence[f"weight{index}"] = dict(shape=list(parameter.shape),
            local_shape=list(parameter.to_local().shape), stride=list(parameter.stride()),
            mesh=parameter.device_mesh.mesh.tolist(), placements=[str(p) for p in parameter.placements])
    return evidence


def _native_worker(rank, directory, world, num_local_experts):
    context = _initialize(rank, directory, world)
    try:
        model, backend, manager, coordinator = prepare_checkpoint_model(context, num_local_experts)
        evidence = _layout_evidence(model, context, num_local_experts)
        optimizer = prepare_checkpoint_optimizer(model, backend)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
        for step in range(2):
            training_step(model, optimizer, context, step)
            scheduler.step()
        checkpoint = str(Path(directory, "checkpoint"))
        coordinator.save_model(model, checkpoint, source_path="/models/source")
        coordinator.save_optimizer_and_scheduler(model, optimizer, scheduler, checkpoint)
        caller = dict(cursor=2, rng=torch.get_rng_state())
        torch.save(caller, Path(checkpoint, f"caller-{rank}.pt"))
        metadata = load_metadata(checkpoint)
        assert metadata["schema_version"] == 2
        layout = metadata["parallel_layout"]
        assert layout["expert_layout_version"] == "gated_logical_v1"
        assert layout["topology"] == dict(world_size=world, sp_size=1, tp_size=2, ep_size=2, etp_size=2)
        entries = {entry["fqn"]: entry for entry in layout["parameters"]}
        assert entries["head.weight"]["aliases"] == ["head.weight", "head_alias.weight"]
        for index in (1, 2):
            entry = entries[f"moe.experts.weight{index}"]
            for lane in entry["layouts"]:
                start = lane["ep_rank"] * num_local_experts
                assert lane["expert_ids"] == list(range(start, start + num_local_experts))
        model_metadata = FileSystemReader(manager.native_model_dir(checkpoint)).read_metadata()
        optimizer_metadata = FileSystemReader(manager.optimizer_dir(checkpoint)).read_metadata()
        for index in (1, 2):
            for ep in (0, 1):
                fqn = f"moe.experts.weight{index}::ep{ep}"
                assert tuple(model_metadata.state_dict_metadata[f"model.{fqn}"].size) == tuple(model.get_parameter(f"moe.experts.weight{index}").shape)
                for field in ("exp_avg", "exp_avg_sq"):
                    assert tuple(optimizer_metadata.state_dict_metadata[f"optimizer.state.{fqn}.{field}"].size) == tuple(model.get_parameter(f"moe.experts.weight{index}").shape)
        keys = list(model_metadata.state_dict_metadata) + list(optimizer_metadata.state_dict_metadata)
        assert all("::tp" not in key and "::etp" not in key for key in keys)
        assert "optimizer.param_groups.0.params" in optimizer_metadata.state_dict_metadata
        restored, restore_backend, _, restore = prepare_checkpoint_model(context, num_local_experts)
        restored_optimizer = prepare_checkpoint_optimizer(restored, restore_backend, lr=.1)
        restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=.8)
        identities = [id(parameter) for parameter in restored.parameters()]
        restore.load_model(restored, checkpoint, source_path="/models/source")
        restore.load_optimizer_and_scheduler(restored, restored_optimizer, restored_scheduler, checkpoint)
        assert [id(parameter) for parameter in restored.parameters()] == identities
        assert restored.head is restored.head_alias
        assert restore_backend._parameter_ownership.is_expert(restored.moe.experts.weight1)
        assert scheduler.state_dict() == restored_scheduler.state_dict()
        assert_exact_model_and_optimizer(model, optimizer, restored, restored_optimizer)
        resumed_caller = torch.load(Path(checkpoint, f"caller-{rank}.pt"), weights_only=True)
        torch.testing.assert_close(resumed_caller["rng"], caller["rng"], rtol=0, atol=0)
        assert resumed_caller["cursor"] == 2
        for current, current_optimizer, current_scheduler, state in (
            (model, optimizer, scheduler, caller),
            (restored, restored_optimizer, restored_scheduler, resumed_caller),
        ):
            torch.set_rng_state(state["rng"])
            offset = torch.rand(()).item()
            training_step(current, current_optimizer, context, state["cursor"] + offset)
            current_scheduler.step()
            state["cursor"] += 1
            state["rng"] = torch.get_rng_state()
        assert scheduler.state_dict() == restored_scheduler.state_dict()
        assert caller["cursor"] == resumed_caller["cursor"] == 3
        torch.testing.assert_close(caller["rng"], resumed_caller["rng"], rtol=0, atol=0)
        assert_exact_model_and_optimizer(model, optimizer, restored, restored_optimizer)
        evidence.update(native_keys=keys, owner_ranks=context.expert_checkpoint_owner_ranks,
                        cursor=3, exact_resume=True)
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(evidence, indent=2))
    finally:
        dist.destroy_process_group()


def _legacy_direct_hf_worker(rank, directory):
    from torchturbo.engine.backends.fsdp2.checkpoint.weight_loader import DistParamLoader
    from torchturbo.engine.backends.fsdp2.checkpoint.weight_saver import ShardedParamManager

    dist.init_process_group("gloo", init_method=f"file://{directory}/store", rank=rank,
                            world_size=4, timeout=timedelta(seconds=90))
    try:
        context = ParallelContext("cpu", sp_size=1, ep_size=2)
        expected_ranks = [0, 2] if rank % 2 == 0 else [1, 3]
        expected = hf_values(CheckpointMoE(context, 1), 2)
        source = Path(directory, "source")
        if rank == 0:
            write_source(source, expected)
        dist.barrier()
        model, backend, _, coordinator = prepare_checkpoint_model(context, 1, meta=True)
        assert backend.gradient_scaling_factor == .5
        converter = coordinator._weight_converter(model, str(source))
        loader = DistParamLoader(
            model=model, path=source, rank=rank, mesh=context.expert_mesh,
            device_runtime=CpuRuntime(), is_cpu_offload=False, weight_converter=converter,
        )
        assert loader.expert_owner_ranks == expected_ranks
        assert dist.get_process_group_ranks(loader.expert_owner_group) == expected_ranks
        for name, owner in loader.param_to_rank.items():
            if ".experts." in name:
                expert_ranks = [0, 2] if name.split(".")[2] == "0" else [1, 3]
                assert owner in expert_ranks
        loader.load_weights(assign=True)
        backend.finalize_model_load(model)
        saver = ShardedParamManager(
            model=model, path=source, rank=rank, mesh=context.expert_mesh,
            device_runtime=CpuRuntime(), weight_converter=converter,
        )
        assert saver.expert_owner_ranks == expected_ranks
        for name, owner in saver.param_to_rank.items():
            if ".experts." in name:
                expert_ranks = [0, 2] if name.split(".")[2] == "0" else [1, 3]
                assert owner in expert_ranks
        export = Path(directory, "export")
        saver.save_model(export)
        actual = read_hf(export)
        assert actual.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_legacy_direct_hf_constructors_preserve_2d_owner_group(tmp_path):
    mp.spawn(_legacy_direct_hf_worker, args=(str(tmp_path),), nprocs=4, join=True)


@pytest.mark.parametrize("world,num_local_experts", [(4, 3), (8, 1), (8, 3), (16, 1)])
def test_native_logical_etp_adamw_resume_and_next_step(tmp_path, world, num_local_experts):
    mp.spawn(_native_worker, args=(str(tmp_path), world, num_local_experts), nprocs=world, join=True)
    assert len(list(tmp_path.glob("rank-*.json"))) == world


def _hf_worker(rank, directory, num_local_experts, format):
    context = _initialize(rank, directory, 8, sp=2)
    try:
        source_model = CheckpointMoE(context, num_local_experts, fused=format != "base")
        expected = hf_values(source_model, num_local_experts * 2, fused=format == "fused")
        source = Path(directory, "source")
        if rank == 0:
            write_source(source, expected)
        dist.barrier()
        model, backend, manager, coordinator = prepare_checkpoint_model(
            context, num_local_experts, meta=True, fused=format != "base", dtype=torch.float64)
        identities = [id(parameter) for parameter in model.parameters()]
        from torchturbo.engine.backends.fsdp2.checkpoint.weight_loader import DistParamLoader
        DistParamLoader._BROADCAST_MEM_BUDGET_CAP = 64
        batches = []
        real_flush = DistParamLoader._flush_pending_tasks
        def flush(loader, tasks, state):
            batches.append([dict(name=task.key, source=task.source,
                group=dist.get_process_group_ranks(task.group) if task.group is not None else list(range(8)),
                shape=list(task.full_tensor.shape), dtype=str(task.full_tensor.dtype)) for task in tasks])
            return real_flush(loader, tasks, state)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(DistParamLoader, "_flush_pending_tasks", flush)
            coordinator.load_pretrained(model, str(source))
        backend.finalize_model_load(model)
        assert identities != [id(parameter) for parameter in model.parameters()]
        assert model.head is model.head_alias
        assert backend._parameter_ownership.is_expert(model.moe.experts.weight1)
        assert context.expert_checkpoint_owner_ranks == ([0, 1, 4, 5] if context.ep_mesh.get_local_rank() == 0 else [2, 3, 6, 7])
        evidence = _layout_evidence(model, context, num_local_experts)
        w1, w2 = expert_values(num_local_experts * 2)
        start = context.ep_mesh.get_local_rank() * num_local_experts
        for parameter, oracle in ((model.moe.experts.weight1, w1[start:start + num_local_experts]),
                                  (model.moe.experts.weight2, w2[start:start + num_local_experts])):
            torch.testing.assert_close(parameter.full_tensor().reshape_as(oracle), oracle.double(), rtol=0, atol=0)
        assert len(batches) > 1
        all_batches = [None] * 8
        dist.all_gather_object(all_batches, batches)
        assert all(all_batches[owner] == batches for owner in context.expert_checkpoint_owner_ranks)
        common = [task for batch in batches for task in batch if task["group"] == list(range(8))]
        for peer_batches in all_batches:
            assert [task for batch in peer_batches for task in batch if task["group"] == list(range(8))] == common
        sources = {task["source"] for batch in batches for task in batch}
        assert len(sources) > 1
        for batch in batches:
            for task in batch:
                assert task["source"] in task["group"]
                assert task["dtype"] == "torch.float32"
                if ".experts." in task["name"]:
                    assert task["group"] == context.expert_checkpoint_owner_ranks
        converter = coordinator._weight_converter(model, str(source))
        from torchturbo.engine.backends.fsdp2.checkpoint.weight_saver import ShardedParamManager
        real_assign = ShardedParamManager._assign_param_to_ranks
        def assign(saver, *args):
            saver.WEIGHT_SHARD_SIZE = 64
            return real_assign(saver, *args)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ShardedParamManager, "_assign_param_to_ranks", assign)
            saver = manager._hf_saver(model, str(source), converter)
        converted = []
        converter_grad_modes = []
        original_w1 = converter.get_local_ep_w1_hf_state_dict
        original_w2 = converter.get_local_ep_w2_hf_state_dict
        def convert_w1(name, tensor):
            converted.append(name)
            converter_grad_modes.append(torch.is_grad_enabled())
            return original_w1(name, tensor)
        def convert_w2(name, tensor):
            converted.append(name)
            converter_grad_modes.append(torch.is_grad_enabled())
            return original_w2(name, tensor)
        converter.get_local_ep_w1_hf_state_dict = convert_w1
        converter.get_local_ep_w2_hf_state_dict = convert_w2
        export = str(Path(directory, "export"))
        assert torch.is_grad_enabled()
        assert all(parameter.requires_grad and parameter.dtype == torch.float64 for parameter in model.parameters())
        saver.save_model(export)
        assert torch.is_grad_enabled()
        assert converter_grad_modes == [False, False]
        assert converted == ["moe.experts.weight1", "moe.experts.weight2"]
        assignments = [None] * 8
        dist.all_gather_object(assignments, list(saver.local_params))
        assert sum(bool(keys) for keys in assignments) > 1
        if format == "fused":
            for key in ("moe.experts.gate_up_proj", "moe.experts.down_proj"):
                assert sum(key in keys for keys in assignments) == 1
        exported = read_hf(export)
        assert exported.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(exported[name], expected[name], rtol=0, atol=0)
        optimizer = prepare_checkpoint_optimizer(model, backend)
        training_step(model, optimizer, context, 0)
        identities = [id(parameter) for parameter in model.parameters()]
        hooks = []
        model.moe.experts.register_forward_hook(lambda *args: hooks.append(True))
        manager.load_hf(model, export, weight_converter=converter, assign=False)
        assert identities == [id(parameter) for parameter in model.parameters()]
        assert set(optimizer.state) == set(model.parameters())
        training_step(model, optimizer, context, 1)
        assert hooks
        evidence.update(owner_ranks=context.expert_checkpoint_owner_ranks,
                        writer_keys=sorted(saver.local_params), format=format, assign_rebound=True,
                        resume_identity=True, converted=converted, broadcast_batches=batches,
                        caller_grad_enabled=True, converter_grad_modes=converter_grad_modes,
                        parameter_dtype="torch.float64", hf_dtype="torch.float32")
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(evidence, indent=2))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("num_local_experts", [1, 3])
@pytest.mark.parametrize("format", ["base", "fused", "fallback"])
def test_logical_hf_load_export_uses_fixed_ep_owner_mesh(tmp_path, num_local_experts, format):
    mp.spawn(_hf_worker, args=(str(tmp_path), num_local_experts, format), nprocs=8, join=True)
    assert len(list(tmp_path.glob("rank-*.json"))) == 8


def _preflight_worker(rank, directory):
    context = _initialize(rank, directory, 4)
    try:
        model, backend, manager, coordinator = prepare_checkpoint_model(context, 2)
        optimizer = prepare_checkpoint_optimizer(model, backend)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        training_step(model, optimizer, context, 0)
        checkpoint = str(Path(directory, "checkpoint"))
        coordinator.save_model(model, checkpoint, source_path="/models/source")
        coordinator.save_optimizer_and_scheduler(model, optimizer, scheduler, checkpoint)
        original = load_metadata(checkpoint)
        model_path = Path(manager.native_model_dir(checkpoint), ".metadata")
        optimizer_path = Path(manager.optimizer_dir(checkpoint), ".metadata")
        model_payload, optimizer_payload = model_path.read_bytes(), optimizer_path.read_bytes()
        modes = ["world_size", "sp_size", "tp_size", "ep_size", "etp_size",
                 "config_sp_size", "config_tp_size", "config_ep_size", "config_etp_size",
                 "layout", "layout_schema", "shape", "ownership", "expert_ids", "ep_rank",
                 "mesh", "placements", "fqn", "aliases", "schema1", "missing",
                 "native_shape", "native_missing_ep"]
        evidence = []
        for mode in modes:
            metadata = copy.deepcopy(original)
            layout = metadata["parallel_layout"]
            entry = next(item for item in layout["parameters"] if item["fqn"] == "moe.experts.weight1")
            if mode in layout["topology"]:
                layout["topology"][mode] += 1
            elif mode.startswith("config_"):
                metadata["engine_config"]["parallel_plan"][mode.removeprefix("config_")] += 1
            elif mode == "layout":
                layout["expert_layout_version"] = "packed_2d_v1"
            elif mode == "layout_schema":
                layout["layout_schema_version"] = 99
            elif mode == "shape":
                entry["shape"] = [6, 8]
            elif mode == "ownership":
                entry["ownership"] = "dense"
            elif mode in ("expert_ids", "ep_rank", "mesh", "placements"):
                entry["layouts"][0][mode] = {"expert_ids": [99, 100], "ep_rank": 99,
                    "mesh": [[0, 2]], "placements": ["S(1)", "S(0)"]}[mode]
            elif mode == "fqn":
                entry["fqn"] = "moe.experts.other_weight"
            elif mode == "aliases":
                entry["aliases"] = ["other_alias"]
            elif mode == "schema1":
                metadata["schema_version"] = 1
                del metadata["parallel_layout"]
            if rank == 0:
                save_metadata(metadata, checkpoint)
                if mode == "missing":
                    (Path(checkpoint) / "torchturbo" / "checkpoint.json").unlink()
                    (Path(checkpoint) / "torchturbo").rmdir()
                model_path.write_bytes(model_payload)
                optimizer_path.write_bytes(optimizer_payload)
                if mode.startswith("native_"):
                    for path, payload in ((model_path, model_payload), (optimizer_path, optimizer_payload)):
                        native = pickle.loads(payload)
                        for key in list(native.state_dict_metadata):
                            if "moe.experts.weight1::ep1" not in key:
                                continue
                            if mode == "native_missing_ep":
                                del native.state_dict_metadata[key]
                            elif not key.endswith(".step"):
                                native.state_dict_metadata[key].size = torch.Size([6, 8])
                        path.write_bytes(pickle.dumps(native))
            dist.barrier()
            for component in ("model", "optimizer"):
                fresh_optimizer = prepare_checkpoint_optimizer(model, backend)
                fresh_scheduler = torch.optim.lr_scheduler.StepLR(fresh_optimizer, step_size=1)
                before_scheduler = copy.deepcopy(fresh_scheduler.state_dict())
                before = {name: value.to_local().clone() for name, value in model.state_dict().items()}
                import torchturbo.engine.backends.fsdp2.checkpoint.dcp as dcp_module
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(dcp_module.dcp, "load", lambda *a, **k: pytest.fail(f"{mode}: DCP load entered before validation"))
                    patch.setattr(dcp_module, "get_optimizer_state_dict", lambda *a, **k: pytest.fail(f"{mode}: optimizer allocation entered before validation"))
                    with pytest.raises((ValueError, RuntimeError), match="layout|topology|HF|metadata|shape|EP ranks|required"):
                        if component == "model":
                            coordinator.load_model(model, checkpoint, source_path="/models/source")
                        else:
                            coordinator.load_optimizer_and_scheduler(model, fresh_optimizer, fresh_scheduler, checkpoint)
                assert not fresh_optimizer.state
                assert fresh_scheduler.state_dict() == before_scheduler
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value.to_local(), before[name], rtol=0, atol=0)
                evidence.append(f"{mode}:{component}")
        Path(directory, f"rank-{rank}.json").write_text(json.dumps({"preflight_rejected": evidence}))
    finally:
        dist.destroy_process_group()


def test_logical_native_mismatch_rejects_before_mutation(tmp_path):
    mp.spawn(_preflight_worker, args=(str(tmp_path),), nprocs=4, join=True)
    assert len(list(tmp_path.glob("rank-*.json"))) == 4


def _migration_worker(rank, directory):
    context = _initialize(rank, directory, 4)
    try:
        legacy_context = ParallelContext("cpu", sp_size=1, ep_size=2)
        source_model = CheckpointMoE(legacy_context, 2)
        expected = hf_values(source_model, 4)
        source = Path(directory, "source")
        if rank == 0:
            write_source(source, expected)
        dist.barrier()
        legacy, legacy_backend, _, legacy_coordinator = prepare_checkpoint_model(legacy_context, 2, meta=True)
        legacy_coordinator.load_pretrained(legacy, str(source))
        legacy_backend.finalize_model_load(legacy)
        native = str(Path(directory, "legacy-native"))
        legacy_coordinator.save_model(legacy, native, source_path=str(source), format=CheckpointFormat.NATIVE)
        assert load_metadata(native)["schema_version"] == 1
        assert "parallel_layout" not in load_metadata(native)
        packed, packed_backend, _, packed_coordinator = prepare_checkpoint_model(legacy_context, 2)
        packed_coordinator.load_model(packed, native, source_path=str(source))
        packed_hf = str(Path(directory, "packed-hf"))
        packed_coordinator.save_model(packed, packed_hf, source_path=str(source), format=CheckpointFormat.HF)
        logical, logical_backend, _, logical_coordinator = prepare_checkpoint_model(context, 2, meta=True)
        logical_coordinator.load_pretrained(logical, str(Path(packed_hf, "huggingface")))
        logical_backend.finalize_model_load(logical)
        logical_optimizer = prepare_checkpoint_optimizer(logical, logical_backend)
        assert not logical_optimizer.state
        logical_hf = str(Path(directory, "logical-hf"))
        logical_coordinator.save_model(logical, logical_hf,
            source_path=str(Path(packed_hf, "huggingface")), format=CheckpointFormat.HF)
        reverse, reverse_backend, _, reverse_coordinator = prepare_checkpoint_model(legacy_context, 2, meta=True)
        reverse_coordinator.load_pretrained(reverse, str(Path(logical_hf, "huggingface")))
        reverse_backend.finalize_model_load(reverse)
        reverse_optimizer = prepare_checkpoint_optimizer(reverse, reverse_backend)
        assert not reverse_optimizer.state
        for name, parameter in legacy.named_parameters():
            torch.testing.assert_close(parameter.full_tensor(), reverse.get_parameter(name).full_tensor(), rtol=0, atol=0)
        for path in (Path(packed_hf, "huggingface"), Path(logical_hf, "huggingface")):
            actual = read_hf(path)
            assert actual.keys() == expected.keys()
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        assert logical.head is logical.head_alias and reverse.head is reverse.head_alias
        caller = {"cursor": 0}
        Path(directory, f"rank-{rank}.json").write_text(json.dumps({
            "weights_only_roundtrip": True, "fresh_optimizer": True, "caller": caller}))
    finally:
        dist.destroy_process_group()


def test_packed_native_to_logical_hf_weights_only_migration(tmp_path):
    mp.spawn(_migration_worker, args=(str(tmp_path),), nprocs=4, join=True)
    assert len(list(tmp_path.glob("rank-*.json"))) == 4


def _missing_source_worker(rank, directory, missing):
    context = _initialize(rank, directory, 8)
    try:
        source = Path(directory, "source")
        expected = hf_values(CheckpointMoE(context, 1), 2)
        if rank == 0:
            write_source(source, expected)
        dist.barrier()
        model, backend, manager, coordinator = prepare_checkpoint_model(context, 1, meta=True)
        converter = coordinator._weight_converter(model, str(source))
        loader = manager._hf_loader(model, str(source), converter)
        key = "moe.experts.0.up_proj.weight"
        owner = loader.param_to_rank[key]
        real_read = loader._load_local_shards
        flushed = []
        real_flush = loader._flush_pending_tasks
        def flush(tasks, state):
            flushed.extend(task.key for task in tasks)
            return real_flush(tasks, state)
        def read():
            values = real_read()
            if rank == owner:
                del values[key]
            return values
        if missing == "file":
            if rank == 0:
                (source / loader.param_to_filename[key]).unlink()
            dist.barrier()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(loader, "_flush_pending_tasks", flush)
            if missing == "key":
                patch.setattr(loader, "_load_local_shards", read)
            with pytest.raises((KeyError, RuntimeError, FileNotFoundError), match="up_proj|No such file"):
                loader.load_weights(assign=True)
        assert "moe.experts.weight1" not in flushed
        assert all(parameter.is_meta for parameter in model.parameters())
        assert not Path(directory, "export", "model.safetensors.index.json").exists()
        Path(directory, f"rank-{rank}.json").write_text(json.dumps({
            "missing": missing, "owner": owner, "rejected_before_expert_broadcast": True}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("missing", ["key", "file"])
def test_logical_owner_source_failure_reaches_every_rank(tmp_path, missing):
    mp.spawn(_missing_source_worker, args=(str(tmp_path), missing), nprocs=8, join=True)
    assert len(list(tmp_path.glob("rank-*.json"))) == 8
