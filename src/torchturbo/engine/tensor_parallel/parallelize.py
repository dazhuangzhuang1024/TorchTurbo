"""Resolve and apply model-authored native tensor parallel styles."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from inspect import signature

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ParallelStyle,
    SequenceParallel,
    parallelize_module,
)

from torchturbo.engine.model_fqn import ModelFQNMap
from torchturbo.utils.fqn import match_fqn

from .expert import GroupedMLPParallel


def register_tensor_parallel_cache_guard(module: nn.Module) -> None:
    """Reject explicit cache requests before a TP module starts computation."""
    forward_signature = signature(module.forward)

    def reject_cache(current: nn.Module, args: tuple, kwargs: dict) -> None:
        arguments = forward_signature.bind_partial(*args, **kwargs).arguments
        # Variadic wrappers keep cache options inside bound kwargs rather than named arguments.
        cache = arguments.get("past_key_values", kwargs.get("past_key_values"))
        use_cache = arguments.get("use_cache", kwargs.get("use_cache"))
        if cache is not None or use_cache:
            raise ValueError("TP does not support cache or cached generation.")

    module.register_forward_pre_hook(reject_cache, with_kwargs=True)


class PrepareModuleMetadata(ParallelStyle):
    """Divide explicitly named module metadata by the computation mesh size.

    The attributes must be positive integer counts divisible by the mesh size.
    Only the matched module changes; its global model configuration stays
    unchanged. Dynamic attribute access here is the explicit metadata API,
    not feature detection or version-dependent fallback.

    Args:
        divide: Names of module attributes to convert to local counts.
    """

    divide: tuple[str, ...]

    def __init__(self, *, divide: tuple[str, ...]) -> None:
        super().__init__()
        self.divide = divide

    def validate(self, module: nn.Module, mesh_size: int) -> None:
        """Validate all named counts without changing the module."""
        for name in self.divide:
            try:
                value = getattr(module, name)
            except AttributeError as error:
                raise ValueError(f"TP metadata attribute {name!r} is missing.") from error
            if type(value) is not int or value < 1 or value % mesh_size:
                raise ValueError(
                    f"TP metadata {name!r} ({value!r}) must be a positive "
                    f"integer divisible by mesh size ({mesh_size})."
                )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self.validate(module, device_mesh.size())
        for name in self.divide:
            setattr(module, name, getattr(module, name) // device_mesh.size())
        return module


class SharedHeadNormParallel(SequenceParallel):
    """Normalize local attention heads with a shared replicated norm weight.

    Native DTensor normalization records partial head-sum gradients. Callers
    must reduce these gradients to replicated TP placement before updating
    the norm weights. This style labels the head axis; it does not shard
    sequence tokens. Norm weights must start with equal full values on every
    TP lane.

    Args:
        head_axis: Axis holding the already-local attention heads.
    """

    def __init__(self, *, head_axis: int = 2) -> None:
        super().__init__(sequence_dim=head_axis, use_local_output=True)


def resolve_tensor_parallel_plan(
    plan: Mapping[str, ParallelStyle], model_fqn_map: ModelFQNMap
) -> dict[str, ParallelStyle]:
    """Resolve each pattern to exact modules, rejecting unmatched/overlapping rules.

    Parent and child targets are independent. Each resolved target receives
    its own fresh style instance, including when one pattern matches many
    modules or the same plan prepares another model.
    """
    resolved: dict[str, ParallelStyle] = {}
    for pattern, style in plan.items():
        if not isinstance(style, ParallelStyle):
            raise TypeError(f"TP pattern {pattern!r} requires a ParallelStyle instance.")
        targets = [
            prepared_fqn
            for model_fqn, prepared_fqn in model_fqn_map.modules.items()
            if match_fqn(pattern, model_fqn)
        ]
        if not targets:
            raise ValueError(f"TP pattern {pattern!r} matched no model modules.")
        for target in targets:
            if target in resolved:
                raise ValueError(f"TP target {target!r} matches multiple styles.")
            resolved[target] = deepcopy(style)
    return resolved


def validate_tensor_parallel_targets(
    model: nn.Module,
    plan: Mapping[str, ParallelStyle],
    *,
    tp_mesh: DeviceMesh,
    etp_mesh: DeviceMesh | None = None,
) -> None:
    """Validate used meshes and all targets before any native TP mutation."""
    for fqn, style in plan.items():
        mesh = etp_mesh if isinstance(style, GroupedMLPParallel) else tp_mesh
        if mesh is not None and mesh.ndim != 1:
            raise ValueError(f"TP target {fqn!r} requires a 1D device mesh.")
        module = model.get_submodule(fqn)
        if isinstance(style, PrepareModuleMetadata):
            style.validate(module, tp_mesh.size())
        elif isinstance(style, GroupedMLPParallel):
            if etp_mesh is None:
                raise ValueError("GroupedMLPParallel requires a nontrivial ETP mesh.")
            style.validate(module, etp_mesh.size())


def apply_tensor_parallel_plan(
    model: nn.Module,
    plan: Mapping[str, ParallelStyle],
    *,
    tp_mesh: DeviceMesh,
    etp_mesh: DeviceMesh | None = None,
) -> nn.Module:
    """Apply exact native styles after validating every metadata/expert target."""
    validate_tensor_parallel_targets(model, plan, tp_mesh=tp_mesh, etp_mesh=etp_mesh)
    for fqn, style in plan.items():
        mesh = etp_mesh if isinstance(style, GroupedMLPParallel) else tp_mesh
        if mesh is None:
            raise ValueError("GroupedMLPParallel requires a nontrivial ETP mesh.")
        parallelize_module(model.get_submodule(fqn), mesh, style)
    return model
