"""Topology and stable parameter ownership for the FSDP2 backend."""

import os
import sys
from datetime import timedelta

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


def _check_group(mesh, expected_groups, rank):
    expected = next(group for group in expected_groups if rank in group)
    assert mesh.mesh.tolist() == expected
    assert dist.get_process_group_ranks(mesh.get_group()) == expected
    value = torch.tensor(rank)
    dist.all_reduce(value, group=mesh.get_group())
    assert value.item() == sum(expected)


def _run_context_test(rank, init_file):
    from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext

    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=12,
        timeout=timedelta(seconds=90),
    )
    try:
        world = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        singletons = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]
        sp_groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        dp_groups = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
        ep_groups = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        edp_groups = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
        expert_groups = {
            1: singletons,
            2: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
            4: ep_groups,
        }
        expert_data_groups = {
            1: world,
            2: [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]],
            4: edp_groups,
        }
        for sp_size, ep_size in ((3, 4), (3, 2), (1, 1), (3, 1), (1, 4)):
            context = ParallelContext("cpu", sp_size=sp_size, ep_size=ep_size)
            assert context.dense_mesh.mesh_dim_names == ("dp",)
            assert context.dense_mesh.get_group() is dist.group.WORLD
            assert context.expert_mesh.mesh.shape == (12 // ep_size, ep_size)
            assert context.expert_mesh.get_local_rank("ep") == rank % ep_size
            assert context.expert_mesh.mesh_dim_names == ("edp", "ep")
            assert context.expert_fsdp_mesh.mesh_dim_names == ("edp",)
            for mesh in (context.data_dp_mesh, context.sp_mesh,
                         context.expert_fsdp_mesh, context.ep_mesh):
                assert mesh._get_root_mesh() is context.root_mesh
            _check_group(context.dense_mesh, world, rank)
            _check_group(context.sp_mesh, sp_groups if sp_size == 3 else singletons, rank)
            _check_group(context.data_dp_mesh, dp_groups if sp_size == 3 else world, rank)
            _check_group(context.ep_mesh, expert_groups[ep_size], rank)
            _check_group(context.expert_fsdp_mesh, expert_data_groups[ep_size], rank)
            assert context.get_dp_info()["dp_rank"] == rank // sp_size
            assert context.get_dp_info()["dp_size"] == 12 // sp_size
            assert context.get_sp_info()["sp_rank"] == rank % sp_size
            assert context.get_sp_info()["sp_size"] == sp_size
            assert context.get_ep_info()["ep_rank"] == rank % ep_size
            assert context.get_ep_info()["ep_size"] == ep_size
            if sp_size == 1:
                assert context.get_dp_info()["dp_group"] is dist.group.WORLD
        context = ParallelContext("cpu", sp_size=3, ep_size=2, tp_size=2, etp_size=2)
        assert context.dense_mesh.mesh_dim_names == ("dp", "tp")
        assert context.expert_mesh.mesh_dim_names == ("edp", "ep", "etp")
        _check_group(context.dense_fsdp_mesh, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], rank)
        _check_group(context.tp_mesh, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], rank)
        _check_group(context.sp_mesh, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]], rank)
        _check_group(context.data_dp_mesh, [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]], rank)
        _check_group(context.ep_mesh, [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11]], rank)
        _check_group(context.etp_mesh, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], rank)
        _check_group(context.expert_fsdp_mesh, [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]], rank)
        assert context.get_dp_info()["dp_rank"] == rank // 6
        assert context.get_tp_info()["tp_rank"] == rank % 2
        assert context.get_etp_info()["etp_rank"] == rank % 2
        assert context.get_sp_info()["sp_rank"] == rank % 6 // 2
        for mesh in (context.dense_fsdp_mesh, context.tp_mesh, context.etp_mesh,
                     context.sp_mesh, context.ep_mesh, context.expert_fsdp_mesh):
            assert mesh._get_root_mesh() is context.root_mesh
        for degrees in (dict(sp_size=3, ep_size=1, tp_size=4, etp_size=1),):
            valid = ParallelContext("cpu", **degrees)
            assert valid.get_dp_info()["dp_size"] == 1
        for degrees in (dict(sp_size=4, ep_size=1, tp_size=2, etp_size=1),
                        dict(sp_size=1, ep_size=4, tp_size=1, etp_size=2),
                        dict(sp_size=1, ep_size=1, tp_size=True, etp_size=1)):
            with pytest.raises(ValueError):
                ParallelContext("cpu", **degrees)
        for degree in (0, 5):
            with pytest.raises(ValueError, match="sp_size"):
                ParallelContext("cpu", sp_size=degree, ep_size=1)
            with pytest.raises(ValueError, match="ep_size"):
                ParallelContext("cpu", sp_size=1, ep_size=degree)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_context_preserves_legacy_groups_on_one_root(tmp_path):
    mp.spawn(_run_context_test, args=(str(tmp_path / "init"),), nprocs=12, join=True)


def test_ownership_retains_all_parameter_aliases_and_rebinds_after_assignment():
    from torchturbo.engine.backends.fsdp2.parallel_context import ParameterOwnership

    model = nn.Module()
    model.dense = nn.Linear(2, 2, bias=False)
    model.expert = nn.Linear(2, 2, bias=False)
    model.expert_alias = model.expert
    old_expert = model.expert.weight
    ownership = ParameterOwnership()
    ownership.register(model, {old_expert})

    assert ownership.expert_fqns(model) == {"expert.weight", "expert_alias.weight"}

    assert ownership.is_expert(old_expert)
    assert not ownership.is_expert(model.dense.weight)

    model.load_state_dict({name: value.clone() for name, value in model.state_dict().items()}, assign=True)
    ownership.bind(model)

    assert model.expert.weight is not old_expert
    assert ownership.is_expert(model.expert.weight)
    assert not ownership.is_expert(old_expert)
    assert not ownership.is_expert(model.dense.weight)
    assert ownership.expert_fqns(model) == {"expert.weight", "expert_alias.weight"}


def test_ownership_registration_is_bound_and_does_not_retain_models():
    import gc
    import weakref

    from torchturbo.engine.backends.fsdp2.parallel_context import ParameterOwnership

    ownership = ParameterOwnership()
    model = nn.Linear(2, 2, bias=False)
    with pytest.raises(KeyError):
        ownership.bind(model)
    with pytest.raises(KeyError):
        ownership.expert_fqns(model)
    ownership.register(model, {model.weight})
    assert ownership.is_expert(model.weight)
    assert ownership.expert_fqns(model) == {"weight"}
    model_ref = weakref.ref(model)
    parameter_ref = weakref.ref(model.weight)
    del model
    gc.collect()
    assert model_ref() is None
    assert parameter_ref() is None


def _run_checkpoint_owners(rank, init_file):
    from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=8, timeout=timedelta(seconds=90))
    try:
        context = ParallelContext("cpu", sp_size=1, ep_size=2, tp_size=2, etp_size=2)
        expected = [0, 1, 4, 5] if rank // 2 % 2 == 0 else [2, 3, 6, 7]
        assert context.expert_checkpoint_owner_ranks == expected
        assert context.expert_checkpoint_owner_mesh.mesh.tolist() == expected
        tensor = torch.tensor(rank)
        dist.all_reduce(tensor, group=context.expert_checkpoint_owner_group)
        assert tensor.item() == sum(expected)
        owner_group = context.expert_checkpoint_owner_group
        assert context.expert_checkpoint_owner_group is owner_group
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_checkpoint_owner_group_includes_separated_edp_and_etp_axes(tmp_path):
    mp.spawn(_run_checkpoint_owners, args=(str(tmp_path / "init"),), nprocs=8, join=True)
