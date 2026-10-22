"""Native TP preparation accepts model mappings and explicit device meshes."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, PrepareModuleInput, RowwiseParallel

from torchturbo.engine.model_fqn import ModelFQNMap, build_model_fqn_map
from torchturbo.engine.tensor_parallel import (
    PrepareModuleMetadata,
    apply_tensor_parallel_plan,
    resolve_tensor_parallel_plan,
    validate_tensor_parallel_targets,
)


class ProjectionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 8
        self.num_key_value_heads = 4
        self.up = nn.Linear(4, 8, bias=False)
        self.down = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.down(self.up(x).relu())


def test_resolve_rejects_unmatched_duplicate_and_non_style_plans():
    model = nn.ModuleList([ProjectionBlock()])
    fqn_map = build_model_fqn_map(model)
    with pytest.raises(ValueError, match="missing.*matched no"):
        resolve_tensor_parallel_plan({"missing": ColwiseParallel()}, fqn_map)
    with pytest.raises(ValueError, match="0.up.*multiple"):
        resolve_tensor_parallel_plan({"*.up": ColwiseParallel(), "0.up": RowwiseParallel()}, fqn_map)
    with pytest.raises(TypeError, match="ParallelStyle"):
        resolve_tensor_parallel_plan({"*.up": "colwise"}, fqn_map)


def test_resolve_preserves_prepared_parent_child_targets_and_fresh_styles():
    plan = {
        "layers.*": PrepareModuleMetadata(divide=("num_heads",)),
        "layers.*.up": ColwiseParallel(),
    }
    fqn_map = ModelFQNMap(modules={
        "layers.0": "wrapped.layers.0",
        "layers.0.up": "wrapped.layers.0.up",
        "layers.1": "wrapped.layers.1",
        "layers.1.up": "wrapped.layers.1.up",
    }, parameters={})
    first = resolve_tensor_parallel_plan(plan, fqn_map)
    second = resolve_tensor_parallel_plan(plan, fqn_map)
    assert set(first) == {
        "wrapped.layers.0", "wrapped.layers.1",
        "wrapped.layers.0.up", "wrapped.layers.1.up",
    }
    assert first["wrapped.layers.0.up"] is not first["wrapped.layers.1.up"]
    for fqn, style in first.items():
        assert style is not second[fqn]
        assert all(style is not original for original in plan.values())


def test_metadata_validates_all_values_before_dividing():
    module = ProjectionBlock()
    module.num_key_value_heads = 3
    style = PrepareModuleMetadata(divide=("num_heads", "num_key_value_heads"))
    with pytest.raises(ValueError, match="num_key_value_heads.*3.*2"):
        style.validate(module, 2)
    assert module.num_heads == 8
    for value in (0, 1.5, True):
        module.num_key_value_heads = value
        with pytest.raises(ValueError, match="num_key_value_heads"):
            style.validate(module, 2)
    with pytest.raises(ValueError, match="missing"):
        PrepareModuleMetadata(divide=("missing",)).validate(module, 2)


def _native_mesh_worker(rank, rendezvous):
    from torchturbo.engine.tensor_parallel.expert import GroupedMLPParallel
    from torchturbo.models.modules.moe.experts import GroupedMLP
    from torchturbo.models.modules.moe.moe_config import MoEConfig

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        tp_mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        etp_mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("etp",))
        plan = {
            "up": ColwiseParallel(),
            "": PrepareModuleMetadata(divide=("num_heads", "num_key_value_heads")),
            "down": RowwiseParallel(input_layouts=Shard(-1)),
        }
        invalid = ProjectionBlock()
        invalid.num_key_value_heads = 3
        original_weight = invalid.up.weight
        with pytest.raises(ValueError, match="num_key_value_heads"):
            apply_tensor_parallel_plan(invalid, plan, tp_mesh=tp_mesh)
        assert invalid.up.weight is original_weight
        assert invalid.num_heads == 8

        torch.manual_seed(13)
        model = ProjectionBlock()
        inputs = torch.arange(12, dtype=torch.float32).view(3, 4) / 12
        expected = model(inputs)
        resolved = resolve_tensor_parallel_plan(plan, build_model_fqn_map(model))
        validate_tensor_parallel_targets(model, resolved, tp_mesh=tp_mesh)
        assert model.num_heads == 8
        assert not isinstance(model.up.weight, DTensor)
        assert apply_tensor_parallel_plan(model, resolved, tp_mesh=tp_mesh) is model
        assert model.num_heads == 4
        assert model.num_key_value_heads == 2
        assert model.up.weight.device_mesh is tp_mesh
        torch.testing.assert_close(model(inputs), expected)

        parent = nn.Sequential(nn.Linear(4, 8, bias=False))
        parent_plan = {
            "": PrepareModuleInput(input_layouts=Replicate(), desired_input_layouts=Replicate()),
            "0": ColwiseParallel(),
        }
        apply_tensor_parallel_plan(parent, parent_plan, tp_mesh=tp_mesh)
        assert parent(inputs).shape == (3, 4)

        mixed = nn.Module()
        mixed.up = nn.Linear(4, 8, bias=False)
        mixed.experts = GroupedMLP(2, False, MoEConfig(
            hidden_size=4, moe_ffn_hidden_size=4, gated_linear_unit=True,
            activation_func=nn.functional.silu, params_dtype=torch.float32,
        ))
        mixed_plan = {"up": ColwiseParallel(), "experts": GroupedMLPParallel()}
        original_weight = mixed.up.weight
        with pytest.raises(ValueError, match="ETP mesh"):
            apply_tensor_parallel_plan(mixed, mixed_plan, tp_mesh=tp_mesh)
        assert mixed.up.weight is original_weight
        assert not isinstance(mixed.experts.weight1, DTensor)
        mixed.experts.config.gated_linear_unit = False
        with pytest.raises(ValueError, match="gated_linear_unit"):
            apply_tensor_parallel_plan(mixed, mixed_plan, tp_mesh=tp_mesh, etp_mesh=etp_mesh)
        assert mixed.up.weight is original_weight
        mixed.experts.config.gated_linear_unit = True
        apply_tensor_parallel_plan(mixed, mixed_plan, tp_mesh=tp_mesh, etp_mesh=etp_mesh)
        assert mixed.up.weight.device_mesh is tp_mesh
        assert mixed.experts.weight1.device_mesh is etp_mesh
        assert mixed.experts.weight1.placements == (Shard(2),)
        assert mixed.experts.weight2.placements == (Shard(1),)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_native_plan_accepts_meshes_and_prevalidates_all_targets(tmp_path):
    mp.spawn(_native_mesh_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True)


def _invalid_mesh_worker(rank, rendezvous, invalid_domain):
    from torchturbo.engine.tensor_parallel import GroupedMLPParallel
    from torchturbo.models.modules.moe.experts import GroupedMLP
    from torchturbo.models.modules.moe.moe_config import MoEConfig

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("parallel",))
        invalid_mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("replica", "shard"))
        tp_mesh = invalid_mesh if invalid_domain == "tp" else mesh
        etp_mesh = invalid_mesh if invalid_domain == "etp" else mesh
        model = nn.Module()
        model.up = nn.Linear(4, 8, bias=False)
        model.experts = GroupedMLP(2, False, MoEConfig(
            hidden_size=4, moe_ffn_hidden_size=4, gated_linear_unit=True,
            activation_func=nn.functional.silu, params_dtype=torch.float32,
        ))
        if invalid_domain == "tp":
            plan = {"experts": GroupedMLPParallel(), "up": ColwiseParallel()}
            valid_target = "experts"
        else:
            plan = {"up": ColwiseParallel(), "experts": GroupedMLPParallel()}
            valid_target = "up"
        original_parameters = dict(model.named_parameters())
        original_hooks = {
            name: (dict(module._forward_pre_hooks), dict(module._forward_hooks))
            for name, module in model.named_modules()
        }
        with pytest.raises((ValueError, RuntimeError)):
            apply_tensor_parallel_plan(model, plan, tp_mesh=tp_mesh, etp_mesh=etp_mesh)
        for name, parameter in model.named_parameters():
            assert parameter is original_parameters[name], name
        for name, module in model.named_modules():
            assert (dict(module._forward_pre_hooks), dict(module._forward_hooks)) == original_hooks[name]

        # An unused mesh does not constrain the styles applied to this model.
        apply_tensor_parallel_plan(
            model, {valid_target: plan[valid_target]}, tp_mesh=tp_mesh, etp_mesh=etp_mesh,
        )
        for parameter in model.get_submodule(valid_target).parameters():
            assert isinstance(parameter, DTensor)
            assert parameter.device_mesh is mesh
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("invalid_domain", ["tp", "etp"])
def test_invalid_used_mesh_rejects_before_any_parameters_or_hooks_change(tmp_path, invalid_domain):
    mp.spawn(_invalid_mesh_worker, args=(str(tmp_path / "store"), invalid_domain),
             nprocs=2, join=True)
