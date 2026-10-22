"""Multi-process tests for EP-unique DCP keys and checkpoint failure handling."""

import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from torchturbo.engine.backends.fsdp2.parallel_context import ParameterOwnership


_HAS_GLOO = dist.is_available() and dist.is_gloo_available()


class _Experts(nn.Module):
    """Expert weights plus a LoRA adapter, both local to one EP rank."""

    def __init__(self) -> None:
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(4, 3))
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 2, bias=False)})


class _MoEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(3, 4, bias=False)
        self.experts = _Experts()


def _build_model(world_mesh, edp_mesh, ep_rank: int, scale: float) -> _MoEModel:
    """Shard dense weights over all ranks and expert weights over edp.

    Expert values encode the EP rank, so state restored on the wrong EP rank
    is visible.
    """
    model = _MoEModel()
    dense = torch.arange(12, dtype=torch.float32).reshape(4, 3) * scale
    model.dense.weight = nn.Parameter(
        distribute_tensor(dense, world_mesh, [Shard(0)])
    )
    expert_params = (
        (model.experts, "weight1", (4, 3), 10.0),
        (model.experts.lora_A["default"], "weight", (2, 3), 20.0),
    )
    for module, name, shape, base in expert_params:
        value = torch.full(shape, (base + ep_rank) * scale)
        setattr(
            module,
            name,
            nn.Parameter(distribute_tensor(value, edp_mesh, [Shard(0)])),
        )
    return model


def _local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def _run_ep_checkpoint_test(
    rank: int, world_size: int, init_file: str, checkpoint_dir: str
) -> None:
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        from torchturbo.engine.backends.fsdp2.checkpoint.dcp import DcpCheckpoint

        world_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("edp", "ep"))
        ep_rank = mesh["ep"].get_local_rank()
        checkpoint = DcpCheckpoint(
            ep_rank=ep_rank, ep_size=mesh["ep"].size(),
            ownership=ParameterOwnership(),
        )

        trained = _build_model(world_mesh, mesh["edp"], ep_rank, scale=1.0)
        checkpoint.ownership.register(trained, set(trained.experts.parameters()))
        assert checkpoint._ep_partitioned_fqns(trained) == {
            "experts.weight1",
            "experts.lora_A.default.weight",
        }
        optimizer = torch.optim.AdamW(trained.parameters(), lr=0.1)
        for param in trained.parameters():
            param.grad = torch.ones_like(param) * (ep_rank + 1)
        optimizer.step()

        model_dir = os.path.join(checkpoint_dir, "model")
        optimizer_dir = os.path.join(checkpoint_dir, "optimizer")
        checkpoint.save_model(trained, model_dir)
        checkpoint.save_optimizer(optimizer, trained, optimizer_dir)

        resumed = _build_model(world_mesh, mesh["edp"], ep_rank, scale=0.0)
        checkpoint.ownership.register(resumed, set(resumed.experts.parameters()))
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.1)
        checkpoint.load_model(resumed, model_dir)
        checkpoint.load_optimizer(resumed_optimizer, resumed, optimizer_dir)

        for name, expected in trained.state_dict().items():
            actual = resumed.state_dict()[name]
            torch.testing.assert_close(
                _local(actual), _local(expected), rtol=0, atol=0, msg=name
            )
        expected_state = get_optimizer_state_dict(trained, optimizer)["state"]
        actual_state = get_optimizer_state_dict(resumed, resumed_optimizer)["state"]
        assert actual_state.keys() == expected_state.keys()
        for name, expected_entry in expected_state.items():
            for key, expected in expected_entry.items():
                torch.testing.assert_close(
                    _local(actual_state[name][key]),
                    _local(expected),
                    rtol=0,
                    atol=0,
                    msg=f"{name}.{key}",
                )

        # A later stage can omit expert adapters while restoring the remaining
        # EP state from the same checkpoint.
        reduced = _build_model(world_mesh, mesh["edp"], ep_rank, scale=0.0)
        del reduced.experts.lora_A
        checkpoint.ownership.register(reduced, set(reduced.experts.parameters()))
        checkpoint.load_model(reduced, model_dir)
        for name, actual in reduced.state_dict().items():
            torch.testing.assert_close(
                _local(actual), _local(trained.state_dict()[name]), rtol=0, atol=0
            )

        from torchturbo.engine.checkpoint.distributed import collective_check, rank_zero_call

        assert rank_zero_call(lambda: rank) == 0

        def missing_on_rank_zero():
            raise FileNotFoundError("rank-zero preflight failure")

        with pytest.raises(FileNotFoundError, match="rank-zero preflight"):
            rank_zero_call(missing_on_rank_zero)

        def missing_on_one_rank():
            if rank == world_size - 1:
                raise FileNotFoundError("rank-local preflight failure")

        with pytest.raises(FileNotFoundError, match="rank-local preflight"):
            collective_check(missing_on_one_rank)

        # A rank with a different filesystem view must fail together with peers
        # before any of them starts the DCP load collective.
        local_model_dir = model_dir if rank != world_size - 1 else model_dir + "-missing"
        with pytest.raises(FileNotFoundError):
            checkpoint.load_model(resumed, local_model_dir)

        from torchturbo.engine.backends.fsdp2.checkpoint.manager import FSDP2CheckpointManager
        from torchturbo.engine.backends.fsdp2.optimizers.memory import OptimizerMemoryController
        from torchturbo.engine.checkpoint.coordinator import CheckpointCoordinator
        from torchturbo.engine.config import EngineConfig
        from torchturbo.runtime.cpu import CpuRuntime

        controller = OptimizerMemoryController(optimizer, CpuRuntime())
        parallel_context = {"ep": {"ep_rank": ep_rank, "ep_size": mesh["ep"].size()}}
        manager = FSDP2CheckpointManager(
            rank=rank,
            parallel_context=SimpleNamespace(
                get_ep_info=lambda: parallel_context["ep"], tp_size=1, etp_size=1,
                expert_mesh=mesh, root_mesh=world_mesh,
            ),
            ownership=checkpoint.ownership,
            device_runtime=CpuRuntime(),
            cpu_offload=False,
            require_optimizer_not_suspended=controller.require_not_suspended,
        )
        coordinator = CheckpointCoordinator(
            EngineConfig(), manager, parallel_context=parallel_context
        )
        coordinator.save_optimizer_and_scheduler(
            trained, optimizer, None, checkpoint_dir
        )
        metadata = Path(manager.optimizer_dir(checkpoint_dir)) / ".metadata"
        saved_metadata = metadata.read_bytes()
        optimizer.zero_grad(set_to_none=True)
        # One suspended rank must stop every peer before any checkpoint I/O.
        if rank == world_size - 1:
            controller.offload()
        with pytest.raises(RuntimeError, match="optimizer is suspended"):
            coordinator.save_optimizer_and_scheduler(
                trained, optimizer, None, checkpoint_dir
            )
        with pytest.raises(RuntimeError, match="optimizer is suspended"):
            coordinator.load_optimizer_and_scheduler(
                trained, optimizer, None, checkpoint_dir
            )
        assert metadata.read_bytes() == saved_metadata
        if rank == world_size - 1:
            controller.restore()
        coordinator.load_optimizer_and_scheduler(
            trained, optimizer, None, checkpoint_dir
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not _HAS_GLOO,
    reason="requires torch.distributed with Gloo support",
)
def test_expert_and_expert_lora_state_round_trip_per_ep_rank(tmp_path):
    mp.spawn(
        _run_ep_checkpoint_test,
        args=(4, str(tmp_path / "init"), str(tmp_path / "checkpoint")),
        nprocs=4,
        join=True,
    )


def _run_shard_read_failure_test(
    rank: int, world_size: int, init_file: str, weights_dir: str
) -> None:
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        from torchturbo.engine.backends.fsdp2.checkpoint import weight_loader
        from torchturbo.runtime.cpu import CpuRuntime

        def load_file(path, device):
            if rank == world_size - 1:
                raise OSError("rank-local shard read failure")
            return {"weight": torch.ones(2)}

        # Spawned workers import their own copy of the module.
        weight_loader.load_file = load_file
        loader = weight_loader.DistParamLoader.__new__(weight_loader.DistParamLoader)
        loader.rank = rank
        loader.path = Path(weights_dir)
        loader.local_params = {"weight"}
        loader.param_to_filename = {"weight": "model.safetensors"}
        loader.param_converter = SimpleNamespace(is_virtual_key=lambda name: False)
        loader.device_runtime = CpuRuntime()
        loader.start_time = loader.prev_time = time.perf_counter()

        # Ranks that read their shards must not wait in a collective that the
        # failed rank never joins.
        with pytest.raises(OSError, match="rank-local shard read failure"):
            loader.load_weights()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not _HAS_GLOO,
    reason="requires torch.distributed with Gloo support",
)
def test_rank_local_shard_read_failure_reaches_every_rank(tmp_path):
    mp.spawn(
        _run_shard_read_failure_test,
        args=(2, str(tmp_path / "init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )


def _run_incomplete_ep_optimizer(rank, init_file, directory):
    import copy
    import torch.distributed.checkpoint as dcp
    from torchturbo.engine.backends.fsdp2.checkpoint.dcp import DcpCheckpoint
    from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=4, timeout=timedelta(seconds=60))
    try:
        context = ParallelContext("cpu", sp_size=1, ep_size=2, tp_size=2, etp_size=2)
        ep_rank = context.get_ep_info()["ep_rank"]
        model = nn.Module()
        model.expert = nn.Linear(3, 4, bias=False)
        model.expert.weight = nn.Parameter(distribute_tensor(
            torch.full((4, 3), ep_rank + 1.0), context.expert_checkpoint_owner_mesh, [Shard(0)]
        ))
        ownership = ParameterOwnership()
        ownership.register(model, {model.expert.weight})
        checkpoint = DcpCheckpoint(ep_rank, 2, ownership, stamp_all_aliases=True)
        optimizer = torch.optim.AdamW(model.parameters())
        model.expert.weight.grad = torch.ones_like(model.expert.weight)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state = checkpoint._stamp_optim_state(
            get_optimizer_state_dict(model, optimizer), checkpoint._ep_partitioned_fqns(model)
        )
        # Keep a valid metadata publication while dropping one consumed EP partition.
        if ep_rank == 1:
            state["state"].clear()
        dcp.save({"optimizer": state}, checkpoint_id=directory)
        for initialized in (False, True):
            current_optimizer = optimizer if initialized else torch.optim.AdamW(model.parameters())
            before = copy.deepcopy(current_optimizer.state_dict())
            parameter_before = model.expert.weight.to_local().clone()
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(dcp, "load", lambda *args, **kwargs: pytest.fail("DCP entered before EP coverage check"))
                with pytest.raises(ValueError, match="missing.*expert.weight::ep1"):
                    checkpoint.load_optimizer(current_optimizer, model, directory)
            actual = current_optimizer.state_dict()
            assert actual["param_groups"] == before["param_groups"]
            assert actual["state"].keys() == before["state"].keys()
            for parameter_id, entry in actual["state"].items():
                for field, tensor in entry.items():
                    torch.testing.assert_close(_local(tensor), _local(before["state"][parameter_id][field]), rtol=0, atol=0)
            torch.testing.assert_close(model.expert.weight.to_local(), parameter_before, rtol=0, atol=0)
            assert model.expert.weight.grad is None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _HAS_GLOO, reason="Gloo unavailable")
def test_optimizer_only_preflight_rejects_missing_ep_partition_before_state_mutation(tmp_path):
    mp.spawn(_run_incomplete_ep_optimizer, args=(str(tmp_path / "init"), str(tmp_path / "optimizer")), nprocs=4, join=True)


def _run_sparse_native_optimizer(rank, init_file, directory):
    import copy
    import json
    import torch.distributed.checkpoint as dcp
    from torch.distributed.fsdp import fully_shard
    from torchturbo.engine.backends.fsdp2.checkpoint.dcp import DcpCheckpoint
    from tests.engine.backends.fsdp2.test_dcp import SparseAdamModel, sparse_optimizer, sparse_step
    from tests.engine.backends.fsdp2.test_tp_checkpoint import _assert_state_equal
    from tests.engine.backends.fsdp2.etp_checkpoint_helpers import assert_tensor_bytes_equal
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    records = []
    try:
        mesh = init_device_mesh("cpu", (2,))
        ownership = ParameterOwnership()
        checkpoint = DcpCheckpoint(0, 1, ownership, stamp_all_aliases=True)

        def prepare(reverse=False):
            torch.manual_seed(91)
            model = SparseAdamModel()
            fully_shard(model.a, mesh=mesh)
            fully_shard(model.b, mesh=mesh)
            fully_shard(model, mesh=mesh)
            ownership.register(model, set())
            return model, sparse_optimizer(model, reverse=reverse)

        def exact(left, left_optimizer, right, right_optimizer):
            _assert_state_equal(left.state_dict(), right.state_dict())
            _assert_state_equal(left_optimizer.state_dict(), right_optimizer.state_dict())
            for name, value in left.state_dict().items():
                assert_tensor_bytes_equal(value, right.state_dict()[name])
            for name, p in left.named_parameters():
                other = right.get_parameter(name)
                assert (p in left_optimizer.state) == (other in right_optimizer.state)
                for field, value in left_optimizer.state.get(p, {}).items():
                    assert_tensor_bytes_equal(value, right_optimizer.state[other][field])

        for mode in ("fresh", "superset", "empty"):
            saved, saved_optimizer = prepare()
            sparse_step(saved, saved_optimizer, leaves=("a", "b") if mode == "empty" else ("a",),
                        update=mode != "empty")
            path = Path(directory) / mode
            checkpoint.save_model(saved, str(path / "model"))
            checkpoint.save_optimizer(saved_optimizer, saved, str(path / "optimizer"))
            current, current_optimizer = prepare()
            if mode == "superset":
                sparse_step(current, current_optimizer, leaves=("a", "b"))
            identities = {name: id(p) for name, p in current.named_parameters()}
            checkpoint.load_model(current, str(path / "model"))
            checkpoint.load_optimizer(current_optimizer, current, str(path / "optimizer"))
            exact(saved, saved_optimizer, current, current_optimizer)
            assert identities == {name: id(p) for name, p in current.named_parameters()}
            assert len(current_optimizer.state) == (0 if mode == "empty" else 1)
            for model, optimizer in ((saved, saved_optimizer), (current, current_optimizer)):
                sparse_step(model, optimizer, leaves=("a", "b"))
                assert optimizer.state[model.b.weight]["step"] == 1
            exact(saved, saved_optimizer, current, current_optimizer)
            records.append(dict(mode=mode, restored_bytes_equal=True, next_step_bytes_equal=True))

        saved, saved_optimizer = prepare()
        sparse_step(saved, saved_optimizer)
        path = Path(directory) / "coverage"
        checkpoint.save_optimizer(saved_optimizer, saved, str(path))
        for mode in ("uncovered", "pending"):
            current, current_optimizer = prepare()
            # Only rank 1 lacks the saved-present skeleton slot. All ranks
            # must fail before DCP; the valid peer must not enter its collective.
            # Deliberate target-state participation must not pair unrelated
            # FSDP units' collectives through rank-divergent model forwards.
            with pytest.MonkeyPatch.context() as patch:
                def unexpected_forward(*args, **kwargs):
                    pytest.fail("Negative target setup entered a model forward")
                patch.setattr(current.a, "forward", unexpected_forward)
                patch.setattr(current.b, "forward", unexpected_forward)
                parameter = current.b.weight if rank else current.a.weight
                parameter.grad = torch.full_like(parameter, .2)
                if mode != "pending" or rank == 0:
                    current_optimizer.step()
            expected_slots = {parameter} if mode != "pending" or rank == 0 else set()
            assert set(current_optimizer.state) == expected_slots
            before = copy.deepcopy(current_optimizer.state_dict())
            model_before = copy.deepcopy(current.state_dict())
            gradients = {name: None if p.grad is None else p.grad.to_local().clone()
                         for name, p in current.named_parameters()}
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(dcp, "load", lambda *args, **kwargs: pytest.fail("DCP entered before collective skeleton check"))
                with pytest.raises(ValueError, match="Native.*skeleton.*a.weight"):
                    checkpoint.load_optimizer(current_optimizer, current, str(path))
            _assert_state_equal(current_optimizer.state_dict(), before)
            _assert_state_equal(current.state_dict(), model_before)
            for name, p in current.named_parameters():
                if gradients[name] is None:
                    assert p.grad is None
                else:
                    torch.testing.assert_close(p.grad.to_local(), gradients[name], rtol=0, atol=0)
            records.append(dict(mode=mode, collective_rejection=True, pre_get_state_preserved=True,
                                target_setup="direct DTensor gradient, no model forward",
                                target_slot_names=[name for name, p in current.named_parameters()
                                                   if p in current_optimizer.state]))

        from torchturbo.engine.backends.fsdp2.checkpoint import dcp as checkpoint_module
        for mode in ("membership", "empty-reordered"):
            saved, saved_optimizer = prepare()
            sparse_step(saved, saved_optimizer, leaves=("a", "b") if mode == "empty-reordered" else ("a",),
                        update=mode != "empty-reordered")
            state = get_optimizer_state_dict(saved, saved_optimizer)
            if mode == "membership":
                state["param_groups"][1]["params"].clear()
            path = Path(directory) / mode
            dcp.save({"optimizer": state}, checkpoint_id=str(path))
            current, current_optimizer = prepare(reverse=mode == "empty-reordered" and rank == 1)
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(checkpoint_module, "set_optimizer_state_dict",
                              lambda *args, **kwargs: pytest.fail("setter entered before collective membership check"))
                match = "Saved.*group.*b.weight" if mode == "membership" else "Empty optimizer state group 0.*b.weight"
                with pytest.raises(ValueError, match=match):
                    checkpoint.load_optimizer(current_optimizer, current, str(path))
            assert len(current_optimizer.state) == 2
            records.append(dict(mode=mode, collective_rejection=True, timing="after native getter and DCP, before setter"))
    finally:
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(records, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(not _HAS_GLOO, reason="Gloo unavailable")
def test_native_sparse_adam_restore_and_collective_failure_boundaries(tmp_path):
    mp.spawn(_run_sparse_native_optimizer, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True)
