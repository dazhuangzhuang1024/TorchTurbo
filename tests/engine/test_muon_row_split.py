"""Independent MLA row blocks in Muon storage and compute layouts."""

import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from torchturbo.engine.backends.fsdp2.optimizers.muon_layout import _MuonLayoutPlanner
from torchturbo.utils.torch_utils import torch_version


def _layout(num_ranks, batch_size=2, row_sizes=(3, 1), columns=5):
    from torchturbo.engine.optimizers import RowSplitComputeView

    shape = torch.Size((batch_size * sum(row_sizes), columns))
    local_rows, _ = Shard.local_shard_size_and_offset(shape[0], num_ranks, 0)
    planner = _MuonLayoutPlanner(tuple(range(num_ranks)), 0)
    layout = planner.resolve_parameter(
        fqn="attention.weight",
        global_storage_shape=shape,
        local_storage_shape=torch.Size((local_rows, columns)),
        compute_view=RowSplitComputeView(batch_size, row_sizes).resolve(shape),
    )
    return planner, layout


def test_row_split_keeps_each_head_and_component_independent():
    _, layout = _layout(3)

    assert [
        (matrix.global_row_offset, matrix.global_column_offset,
         matrix.num_rows, matrix.num_columns)
        for matrix in layout.matrices
    ] == [(0, 0, 3, 5), (3, 0, 1, 5), (4, 0, 3, 5), (7, 0, 1, 5)]


@pytest.mark.parametrize("num_ranks", [1, 3, 12])
def test_row_split_transfers_cover_storage_and_restore_every_element(num_ranks):
    planner, layout = _layout(num_ranks)
    # Rank zero owns unequal shapes; other ranks may store only part of a block.
    owners = (0, 0, num_ranks - 1, num_ranks - 1)
    plan = planner.build_plan(layout, owners)
    original = torch.arange(40).reshape(8, 5)
    storage = tuple(
        original[part.global_row_offset:part.global_row_end].clone()
        for part in plan.storage_partitions
    )
    compute = tuple(torch.full(part.tensor_shape, -1) for part in plan.compute_partitions)

    def transfer(sources, destinations, routes):
        coverage = [torch.zeros_like(value) for value in destinations]
        for route in routes:
            shape = (route.num_rows, route.num_columns)
            src = sources[route.source_group_rank].reshape(-1).as_strided(
                shape, (route.source_row_stride, 1), route.source_flat_offset
            )
            dst = destinations[route.destination_group_rank].reshape(-1).as_strided(
                shape, (route.destination_row_stride, 1), route.destination_flat_offset
            )
            dst.copy_(src)
            coverage[route.destination_group_rank].reshape(-1).as_strided(
                shape, (route.destination_row_stride, 1), route.destination_flat_offset
            ).add_(1)
        assert all(torch.all(value == 1) for value in coverage)

    transfer(storage, compute, plan.storage_to_compute_routes)
    for partition, value in zip(plan.compute_partitions, compute, strict=True):
        expected = torch.cat([
            original[m.global_row_offset:m.global_row_offset + m.num_rows].reshape(-1)
            for index in partition.matrix_indices
            for m in (plan.matrices[index],)
        ]) if partition.matrix_indices else torch.empty(0, dtype=original.dtype)
        torch.testing.assert_close(value.reshape(-1), expected)
        assert sum(group.tensor_shape.numel() for group in partition.matrix_groups) == value.numel()
    restored = tuple(torch.full_like(value, -1) for value in storage)
    transfer(compute, restored, plan.compute_to_storage_routes)
    torch.testing.assert_close(torch.cat(restored), original)


@pytest.mark.parametrize("batch_size,row_sizes,shape", [
    (0, (3, 1), (8, 5)),
    (2, (), (8, 5)),
    (2, (3, 0), (6, 5)),
    (2, (3, -1), (4, 5)),
    (2, (3, 1), (7, 5)),
    (2, (3, 1), (8, 0)),
])
def test_row_split_rejects_incomplete_or_empty_matrix_partitions(batch_size, row_sizes, shape):
    from torchturbo.engine.optimizers import RowSplitComputeView

    with pytest.raises(ValueError):
        RowSplitComputeView(batch_size, row_sizes).resolve(torch.Size(shape))


@pytest.mark.parametrize("policy,expected", [
    (None, (2.0, 1.0)),
    ("original", (2.0, 1.0)),
    ("match_rms_adamw", (0.2 * 8**0.5, 0.2 * 8**0.5)),
    ("spectral_unclamped", (2.0, 0.5)),
])
def test_matrix_scaling_uses_each_component_shape(policy, expected):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon

    optimizer = SimpleNamespace(_group=lambda item: {"adjust_lr_fn": policy})
    for shape, ratio in zip(((3, 8, 2), (3, 2, 8)), expected, strict=True):
        actual = FSDPDistributedMuon._matrix_lr_scale(optimizer, None, shape)
        assert actual == pytest.approx(ratio)


def _run_row_split_steps(
    rank, world_size, init_file, offload_mode, bucket_strategy, execution, adjust_lr_fn,
    dtype=torch.float32, reverse_mesh=False,
):
    from torchturbo.engine.config import MuonOptimizerConfig
    from torchturbo.engine.optimizers import (
        MuonParameterSpec, MuonPlan, MuonRecipe, Owned, RowSplitComputeView,
    )
    from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon
    from torchturbo.runtime.cpu import CpuRuntime
    from torchturbo.engine.backends.fsdp2.config import CPUOffloadConfig
    from tests.engine.test_fsdp_muon_distributed import _StubStreamCpuRuntime

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = (
            DeviceMesh("cpu", list(reversed(range(world_size))), mesh_dim_names=("dp",))
            if reverse_mesh
            else init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        )
        mesh_index = mesh.get_coordinate()[0]

        def distribute(value):
            if not reverse_mesh:
                return distribute_tensor(value, mesh, [Shard(0)])
            # PyTorch 2.10 scatter/gather uses group order for permuted meshes.
            # Build coordinate-ordered storage explicitly and compare local shards.
            local_rows, offset = Shard.local_shard_size_and_offset(
                value.shape[0], world_size, mesh_index
            )
            return DTensor.from_local(
                value[offset:offset + local_rows].clone(), mesh, [Shard(0)],
                shape=value.shape, stride=value.stride(),
            )

        generator = torch.Generator().manual_seed(971)
        layouts = {
            "q": (2, (6, 2), 3),
            "kv": (2, (6, 4), 3),
            "kv_down": (1, (5, 2), 4),
            "dense": (1, (7,), 3),
            "tiny": (1, (1, 1), 3),
        }
        model = nn.Module()
        specs = {}
        references = {}
        for name, (heads, rows, columns) in layouts.items():
            value = torch.randn(heads * sum(rows), columns, generator=generator)
            if dtype == torch.bfloat16:
                # Zero weights expose rounding of the update itself.
                value = torch.zeros_like(value, dtype=dtype)
            model.register_parameter(name, nn.Parameter(distribute(value)))
            specs[name] = (
                MuonParameterSpec(compute_placement=Owned()) if name == "dense"
                else MuonParameterSpec(
                    compute_placement=Shard(0),
                    compute_view=RowSplitComputeView(heads, rows),
                )
            )
            references[name] = [
                nn.Parameter(matrix.clone())
                for head in value.chunk(heads)
                for matrix in head.split(rows)
            ]
        config = MuonOptimizerConfig(
            lr=0.03, weight_decay=0.2, momentum=0.8, ns_steps=3,
            adjust_lr_fn=adjust_lr_fn, bucket={"strategy": bucket_strategy},
            overlap_comm_compute=execution == "async-overlap",
        )
        plan = MuonPlan(model, recipe=MuonRecipe(specs), config=config)
        offload = CPUOffloadConfig(
            enabled=offload_mode is not None, mode=offload_mode or "full",
            optimizer_memory_fraction=0,
        )
        runtime = CpuRuntime() if execution == "sync" else _StubStreamCpuRuntime()
        optimizer = FSDPDistributedMuon(plan, device_runtime=runtime, cpu_offload=offload)
        assert optimizer._workspace.async_enabled == (execution != "sync")
        reference = torch.optim.Muon(
            [matrix for matrices in references.values() for matrix in matrices],
            **plan.muon_kwargs,
        )
        for step in range(3):
            for name, (heads, rows, columns) in layouts.items():
                gradient = torch.randn(heads * sum(rows), columns, generator=generator)
                gradient = gradient.to(dtype)
                model.get_parameter(name).grad = distribute(gradient)
                slices = [matrix for head in gradient.chunk(heads) for matrix in head.split(rows)]
                for parameter, matrix in zip(references[name], slices, strict=True):
                    parameter.grad = matrix.clone()
            optimizer.step()
            reference.step()
            for name, parameter in model.named_parameters():
                expected = torch.cat(references[name])
                local_rows, offset = Shard.local_shard_size_and_offset(
                    expected.shape[0], world_size, mesh_index
                )
                torch.testing.assert_close(
                    parameter.to_local(), expected[offset:offset + local_rows],
                    atol=1e-7, rtol=1e-6,
                )
                momentum = optimizer.state[parameter]["momentum_buffer"]
                expected_momentum = torch.cat([
                    reference.state[matrix]["momentum_buffer"]
                    for matrix in references[name]
                ])
                if offload_mode is None:
                    momentum = momentum.to_local()
                torch.testing.assert_close(
                    momentum, expected_momentum[offset:offset + local_rows], atol=0, rtol=0
                )
            if step == 0:
                state = copy.deepcopy(optimizer.state_dict())
                optimizer = FSDPDistributedMuon(plan, device_runtime=runtime, cpu_offload=offload)
                optimizer.load_state_dict(state)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available() or torch_version() < (2, 10, 0),
    reason="requires Gloo and PyTorch 2.10",
)
@pytest.mark.parametrize("world_size,offload_mode,bucket_strategy,execution,adjust_lr_fn", [
    (1, None, "single", "sync", "match_rms_adamw"),
    (3, None, "single", "sync", "match_rms_adamw"),
    (3, None, "parameter", "sync", "match_rms_adamw"),
    (3, "optimizer", "single", "sync", "match_rms_adamw"),
    (3, "full", "single", "sync", "match_rms_adamw"),
    (3, None, "parameter", "async-overlap", None),
    (3, None, "single", "async", "match_rms_adamw"),
])
def test_row_split_steps_and_resume_match_independent_muon(
    tmp_path, world_size, offload_mode, bucket_strategy, execution, adjust_lr_fn
):
    # Isolate process groups and Optimizer's class-level step-hook wrapping.
    torch.multiprocessing.spawn(
        _run_row_split_steps, nprocs=world_size,
        args=(world_size, str(tmp_path / "init"), offload_mode, bucket_strategy,
              execution, adjust_lr_fn),
        join=True,
    )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available() or torch_version() < (2, 10, 0),
    reason="requires Gloo and PyTorch 2.10",
)
@pytest.mark.parametrize("world_size,reverse_mesh", [(1, False), (3, False), (3, True)])
def test_bfloat16_row_split_updates_match_independent_muon(tmp_path, world_size, reverse_mesh):
    torch.multiprocessing.spawn(
        _run_row_split_steps, nprocs=world_size,
        args=(world_size, str(tmp_path / "init"), None, "single", "sync",
              "match_rms_adamw", torch.bfloat16, reverse_mesh),
        join=True,
    )


def _run_mixed_dtype_local_steps(rank, world_size, init_file, bucket_strategy):
    from torchturbo.engine.backends.fsdp2.config import CPUOffloadConfig
    from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon
    from torchturbo.engine.config import MuonOptimizerConfig
    from torchturbo.engine.optimizers import (
        MuonParameterSpec, MuonPlan, MuonRecipe, RowSplitComputeView,
    )
    from torchturbo.runtime.cpu import CpuRuntime

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        heads = 2 * world_size
        model = nn.Module()
        references = {}
        specs = {}
        for name, dtype in (("q", torch.float32), ("kv", torch.bfloat16)):
            value = torch.zeros(heads * 4, 4, dtype=dtype)
            model.register_parameter(
                name, nn.Parameter(distribute_tensor(value, mesh, [Shard(0)]))
            )
            specs[name] = MuonParameterSpec(
                compute_placement=Shard(0),
                compute_view=RowSplitComputeView(heads, (3, 1)),
            )
            references[name] = [
                nn.Parameter(matrix.clone())
                for head in value.chunk(heads)
                for matrix in head.split((3, 1))
            ]
        plan = MuonPlan(
            model, recipe=MuonRecipe(specs),
            config=MuonOptimizerConfig(
                lr=0.01, weight_decay=0.1, ns_steps=2,
                bucket={"strategy": bucket_strategy, "bucket_cap_mb": 1},
            ),
        )
        optimizer = FSDPDistributedMuon(
            plan, device_runtime=CpuRuntime(), cpu_offload=CPUOffloadConfig()
        )
        reference = torch.optim.Muon(
            [matrix for matrices in references.values() for matrix in matrices],
            **plan.muon_kwargs,
        )
        generator = torch.Generator().manual_seed(534)
        for _ in range(3):
            for name, parameter in model.named_parameters():
                gradient = torch.randn(heads * 4, 4, generator=generator).to(parameter.dtype)
                parameter.grad = distribute_tensor(gradient, mesh, [Shard(0)])
                matrices = [matrix for head in gradient.chunk(heads) for matrix in head.split((3, 1))]
                for matrix, grad in zip(references[name], matrices, strict=True):
                    matrix.grad = grad.clone()
            optimizer.step()
            reference.step()
            for name, parameter in model.named_parameters():
                expected = torch.cat(references[name]).chunk(world_size)[rank]
                torch.testing.assert_close(parameter.to_local(), expected, rtol=1e-6, atol=1e-7)
                momentum = torch.cat([
                    reference.state[matrix]["momentum_buffer"] for matrix in references[name]
                ]).chunk(world_size)[rank]
                torch.testing.assert_close(
                    optimizer.state[parameter]["momentum_buffer"].to_local(),
                    momentum, rtol=0, atol=0,
                )
        assert not optimizer._workspace.slots
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available() or torch_version() < (2, 10, 0),
    reason="requires Gloo and PyTorch 2.10",
)
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("bucket_strategy", ["single", "size"])
def test_local_row_split_supports_mixed_dtype_buckets(tmp_path, world_size, bucket_strategy):
    torch.multiprocessing.spawn(
        _run_mixed_dtype_local_steps, nprocs=world_size,
        args=(world_size, str(tmp_path / "init"), bucket_strategy), join=True,
    )


def test_local_row_split_does_not_overwrite_concurrent_remote_preparation():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon, _CompiledMuonBucket, _MuonWorkspace,
    )
    from torchturbo.engine.optimizers import RowSplitComputeView
    from torchturbo.runtime.cpu import CpuRuntime

    group = SimpleNamespace(
        global_ranks_in_group_order=(0, 1), mesh_indices_in_group_order=(0, 1),
        current_group_rank=0,
    )
    planner = _MuonLayoutPlanner((0, 1), 0)

    def entry(name, shape, view, owners):
        rows, _ = Shard.local_shard_size_and_offset(shape[0], 2, 0)
        local_shape = torch.Size((rows, shape[1]))
        item = SimpleNamespace(
            fqn=name, storage_layout=SimpleNamespace(
                local_shape=local_shape, dtype=torch.float32, device=torch.device("cpu"),
            ),
        )
        layout = planner.resolve_parameter(
            fqn=name, global_storage_shape=torch.Size(shape),
            local_storage_shape=local_shape, compute_view=view,
        )
        return item, planner.build_plan(layout, owners)

    local, local_plan = entry(
        "local", (16, 4), RowSplitComputeView(4, (3, 1)).resolve(torch.Size((16, 4))),
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    remote, remote_plan = entry("remote", (5, 4), None, (0,))
    bucket = _CompiledMuonBucket.compile(
        name="mixed", items=(local, remote),
        parameter_plans=(local_plan, remote_plan), parameter_groups=(group, group),
    )
    workspace = _MuonWorkspace(CpuRuntime(), overlap_comm_compute=True)
    workspace.configure((bucket,))
    buffers = workspace.slot_for_bucket(0).buffers
    updates = []
    optimizer = SimpleNamespace(
        _compute_partition_update=lambda item, prepared, groups: prepared.mul_(2),
        _apply_update=lambda item, direction: updates.append(direction.clone()),
    )

    def prepare(item, prepared):
        prepared.fill_(10 if item is remote else 20)
        if item is remote:
            # With overlap, caller-side computation can finish between remote
            # gradient preparation and transfer-stream packing.
            FSDPDistributedMuon._compute_local_items(optimizer, bucket, buffers)

    optimizer._prepare_local = prepare
    FSDPDistributedMuon._prepare_redistributed(optimizer, bucket, buffers)
    storage, _ = buffers.exchange_buffers(bucket)
    torch.testing.assert_close(storage, torch.full_like(storage, 10))
    torch.testing.assert_close(updates[0], torch.full((8, 4), 40.0))
