"""FSDP-specific parallel plan and module specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from torch import nn
from torch.distributed.tensor.parallel import ParallelStyle

from ..utils.fqn import match_fqn
from .parallel import ParallelPlan

if TYPE_CHECKING:
    from ..models.modules.vision.io import BaseVisionBalanceIO


class VisionBalanceMode(str, Enum):
    """Weight used to distribute whole media across vision encoder ranks."""

    PATCH = "patch"
    PATCH_SQ = "patch_sq"


@dataclass
class VisionBalanceSpec:
    """Configure media balancing and the model's input/output protocol.

    Attributes:
        mode: Media balancing weight, or None to leave balancing disabled in
            the resolved plan. None inherits the model default in an override.
        io: Model-provided input/output adapter. None inherits the model
            adapter in a partial plan override.
    """

    mode: VisionBalanceMode | None = None
    io: BaseVisionBalanceIO | None = None

    def __post_init__(self) -> None:
        if self.mode is not None and not isinstance(self.mode, VisionBalanceMode):
            raise TypeError("VisionBalanceSpec.mode must be a VisionBalanceMode.")


@dataclass
class FSDPPlacementSpec:
    """A placement definition optionally tied to a specific mesh dimension for FSDP.

    Attributes
    ----------
    placement:
        The DTensor placement (e.g., ``Shard(0)``, ``Replicate()``).
    mesh_dim:
        Name or index of the sharding mesh dimension (e.g., ``"ep"``).
        If ``None``, the backend decides the default dimension.
    """

    placement: Any
    mesh_dim: str | int | None = None


@dataclass
class FSDPPlanSpec:
    """Unified logical settings for an FSDP module or its layers.

    FSDP patterns select the first matching spec in insertion order. TP
    styles use a separate map and reject overlapping matches.

    Attributes
    ----------
    placement:
        The parameter/tensor placement definition.
    no_shard:
        Whether to skip FSDP sharding for this module completely.
    standalone_world_shard:
        Whether to shard this module as a standalone FSDP unit on the world
        mesh. With TP enabled, the unit uses its dense FSDP domain.
    activation_checkpoint:
        Whether to apply whole-module activation checkpointing to this module.
    replace_module:
        A callable to replace the matched module. Signature: `(original_module, **kwargs) -> nn.Module`
    module_kwargs:
        Keyword arguments to pass to the `replace_module` callable.
    vision_balance:
        Media balancing mode and the model's input/output adapter. Hooks are
        installed only when the resolved spec has a mode. Non-None fields in
        a partial spec override the corresponding model defaults.
    """

    placement: FSDPPlacementSpec | None = None
    no_shard: bool = False
    standalone_world_shard: bool = False
    activation_checkpoint: bool = False
    replace_module: Any | None = None
    module_kwargs: dict[str, Any] = field(default_factory=dict)
    vision_balance: VisionBalanceSpec | None = None

    def __post_init__(self) -> None:
        if self.vision_balance is not None and not isinstance(
            self.vision_balance, VisionBalanceSpec
        ):
            raise TypeError("FSDPPlanSpec.vision_balance must be a VisionBalanceSpec.")
        if self.no_shard and (
            self.standalone_world_shard or self.placement is not None
        ):
            raise ValueError(
                "FSDPPlanSpec.no_shard is mutually exclusive with "
                "standalone_world_shard and placement."
            )
        if self.standalone_world_shard and self.placement is not None:
            raise ValueError(
                "FSDPPlanSpec.standalone_world_shard is redundant with placement; "
                "use placement for custom mesh sharding."
            )


class TensorParallelValidator(ABC):
    """Validate original and prepared modules without mutation.

    The FSDP2 backend invokes the same validator before mesh creation/module
    replacement and after replacement/target resolution, before TP application.
    Implementations must support meta tensors and inspect live module metadata
    and shapes without changing modules, parameters, hooks, config or groups.
    The parallel plan contains the final merged SP/EP/TP/ETP degrees. Backend
    feature and target admission remains mandatory with a custom validator.
    """

    @abstractmethod
    def validate(self, model: nn.Module, parallel_plan: FSDPParallelPlan) -> None:
        """Raise ValueError when the model does not support the plan."""


@dataclass
class TensorParallelPlanSpec:
    """Declare native TP styles and an optional model contract validator.

    Attributes:
        plan: FQN patterns mapped to native ParallelStyle instances.
        validator: Model validator applied to the resolved parallel plan.
    """

    plan: dict[str, ParallelStyle] = field(default_factory=dict)
    validator: TensorParallelValidator | None = None


@dataclass
class FSDPParallelPlan(ParallelPlan):
    """FSDP-specific parallel plan spanning topologies and placements.

    Attributes
    ----------
    plan:
        FQN pattern to FSDPPlanSpec mapping dictionary.
    sp_size:
        Sequence Parallelism group size.
    ep_size:
        Expert Parallelism group size.
    tp_size:
        Tensor Parallelism group size.
    etp_size:
        Expert Tensor Parallelism group size, configured separately from tp_size
        and defaulting to 1. FSDP2 requires etp_size=1 for dense TP or aligned
        tp_size=etp_size>1 for routed TP.
    tensor_parallel_plan:
        Native TP styles and optional model contract validator. Every target
        must match exactly one style in the spec's plan.
    """

    plan: dict[str, FSDPPlacementSpec | FSDPPlanSpec | Any] = field(
        default_factory=dict
    )
    sp_size: int = 1
    ep_size: int = 1
    tp_size: int = 1
    etp_size: int = 1
    tensor_parallel_plan: TensorParallelPlanSpec = field(
        default_factory=TensorParallelPlanSpec
    )

    def __post_init__(self) -> None:
        """Normalize plan values to FSDPPlanSpec/FSDPPlacementSpec."""
        for k, v in list(self.plan.items()):
            if isinstance(v, tuple) and len(v) == 2:
                self.plan[k] = FSDPPlanSpec(
                    placement=FSDPPlacementSpec(placement=v[0], mesh_dim=v[1])
                )
            elif isinstance(v, FSDPPlacementSpec):
                self.plan[k] = FSDPPlanSpec(placement=v)
            elif not isinstance(v, FSDPPlanSpec):
                self.plan[k] = FSDPPlanSpec(placement=FSDPPlacementSpec(placement=v))

    def match(self, fqn: str) -> Any | None:
        """Find the specification for a given FQN. Returns None if no match."""
        for pattern, spec in self.plan.items():
            if match_fqn(pattern, fqn):
                return spec
        return None
