# The compute-view and bucket abstractions follow the approach taken by
# TorchTitan's DistributedMuon.

"""Backend-independent Muon configuration and matrix views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.distributed.tensor import Shard
from torch import nn

from ...logging import get_logger
from ..model_fqn import ModelFQNMap, build_model_fqn_map
from ...utils.fqn import match_fqn
from .plan import (
    OptimizerPlan,
    OptimizerRecipe,
    ParamGroup,
)
from .qk_clip import QKClipPlan, QKClipSpec

if TYPE_CHECKING:
    from ..config import MuonOptimizerConfig


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Owned:
    """Compute a complete matrix temporarily on one rank of its FSDP mesh."""


@dataclass(frozen=True, slots=True)
class MuonBucket:
    """An ordered bucket whose exact parameter membership is already resolved."""

    name: str
    parameter_fqns: tuple[str, ...]


def _build_muon_buckets(
    parameter_fqns: tuple[str, ...],
    *,
    strategy: str,
    bucket_cap_mb: float | None,
    parameter_num_bytes: Mapping[str, int],
) -> tuple[MuonBucket, ...]:
    if not parameter_fqns:
        return ()

    groups: tuple[tuple[str, ...], ...]
    if strategy == "parameter":
        groups = tuple((fqn,) for fqn in parameter_fqns)
    elif strategy == "single":
        groups = (parameter_fqns,)
    elif strategy == "size":
        if bucket_cap_mb is None:
            raise ValueError("size bucket strategy requires bucket_cap_mb")
        cap_bytes = int(bucket_cap_mb * 1024 * 1024)
        packed: list[tuple[str, ...]] = []
        current: list[str] = []
        current_bytes = 0
        for fqn in parameter_fqns:
            parameter_bytes = parameter_num_bytes[fqn]
            if current and current_bytes + parameter_bytes > cap_bytes:
                packed.append(tuple(current))
                current = []
                current_bytes = 0
            current.append(fqn)
            current_bytes += parameter_bytes
        if current:
            packed.append(tuple(current))
        groups = tuple(packed)
    else:
        raise ValueError(f"unsupported optimizer bucket strategy {strategy!r}")

    return tuple(
        MuonBucket(
            name=f"bucket.{index}",
            parameter_fqns=group,
        )
        for index, group in enumerate(groups)
    )


@dataclass(frozen=True, slots=True)
class ResolvedBatchedMatrixView:
    """Resolved storage-order batch and Muon logical-matrix interpretation."""

    batch_size: int
    matrix_rows: int
    storage_columns: int
    column_groups: int = 1

    @property
    def matrix_columns(self) -> int:
        return self.storage_columns // self.column_groups

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return self.matrix_rows, self.matrix_columns

    @property
    def global_compute_shape(self) -> torch.Size:
        # This is the storage-order batch shape. Independent column groups
        # have matrix_columns columns and may be assigned to different ranks.
        return torch.Size((self.batch_size, self.matrix_rows, self.storage_columns))

    def local_compute_shape(self, storage_shape: torch.Size) -> torch.Size:
        if not (
            len(storage_shape) == 2
            and storage_shape[0] % self.matrix_rows == 0
            and storage_shape[1] == self.storage_columns
        ):
            raise RuntimeError(
                "prepared batched-matrix view is internally inconsistent"
            )
        return torch.Size(
            (
                storage_shape[0] // self.matrix_rows,
                self.matrix_rows,
                self.storage_columns,
            )
        )

    def logical_view(self, compute: Tensor) -> Tensor:
        """Convert a storage-order batch to independent logical matrices.

        ``column_groups > 1`` handles fused projections such as gate/up stored
        as ``[E, H, 2F]``. Reshaping the permutation may materialize a copy;
        restore_storage_order writes logical results back to storage order.
        """
        if self.column_groups == 1:
            return compute
        batch = compute.shape[0]
        grouped = compute.view(
            batch,
            self.matrix_rows,
            self.column_groups,
            self.matrix_columns,
        )
        return grouped.permute(0, 2, 1, 3).reshape(
            batch * self.column_groups,
            self.matrix_rows,
            self.matrix_columns,
        )

    def restore_storage_order(self, logical: Tensor, out: Tensor) -> Tensor:
        if self.column_groups == 1:
            out.copy_(logical)
            return out
        batch = out.shape[0]
        restored = logical.view(
            batch,
            self.column_groups,
            self.matrix_rows,
            self.matrix_columns,
        ).permute(0, 2, 1, 3)
        out.copy_(restored.reshape_as(out))
        return out

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            "batched_matrix",
            self.batch_size,
            self.matrix_rows,
            self.storage_columns,
            self.column_groups,
        )


@runtime_checkable
class MatrixComputeView(Protocol):
    def resolve(self, storage_shape: torch.Size) -> ResolvedBatchedMatrixView: ...


@dataclass(frozen=True, slots=True)
class BatchedMatrixComputeView:
    """Interpret ``[B * R, C]`` storage as a batch of Muon matrices.

    ``column_groups`` splits a fused final dimension into independent matrices.
    For example, ``column_groups=2`` maps ``[E * H, 2F]`` to ``2E`` logical
    matrices of shape ``[H, F]`` while preserving storage-order transport.
    """

    batch_size: int
    column_groups: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(self.column_groups, bool) or self.column_groups <= 0:
            raise ValueError("column_groups must be a positive integer")

    def resolve(self, storage_shape: torch.Size) -> ResolvedBatchedMatrixView:
        if (
            len(storage_shape) != 2
            or storage_shape[0] == 0
            or storage_shape[0] % self.batch_size
            or storage_shape[1] % self.column_groups
        ):
            raise ValueError(
                f"storage shape {tuple(storage_shape)} cannot be interpreted as "
                f"{self.batch_size} matrix batches with "
                f"{self.column_groups} column groups"
            )
        return ResolvedBatchedMatrixView(
            batch_size=self.batch_size,
            matrix_rows=storage_shape[0] // self.batch_size,
            storage_columns=storage_shape[1],
            column_groups=self.column_groups,
        )


@dataclass(frozen=True, slots=True)
class AttentionPerHeadComputeView:
    """Treat ``[H * R, C]`` storage as one Muon matrix per head."""

    num_heads: int

    def __post_init__(self) -> None:
        if isinstance(self.num_heads, bool) or self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")

    def resolve(self, storage_shape: torch.Size) -> ResolvedBatchedMatrixView:
        return BatchedMatrixComputeView(self.num_heads).resolve(storage_shape)


@dataclass(frozen=True, slots=True)
class ExpertUpComputeView:
    """Interpret expert up projections, optionally fused with gate projections.

    Args:
        num_experts: Experts in this parameter after expert parallelism.
        fused_gate_up: Whether each expert's gate and up projections occupy
            separate halves of the storage columns. Each half is an independent
            Muon matrix.
    """

    num_experts: int
    fused_gate_up: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_experts, bool)
            or not isinstance(self.num_experts, int)
            or self.num_experts <= 0
        ):
            raise ValueError("num_experts must be a positive integer")
        if not isinstance(self.fused_gate_up, bool):
            raise TypeError("fused_gate_up must be a bool")

    def resolve(self, storage_shape: torch.Size) -> ResolvedBatchedMatrixView:
        return BatchedMatrixComputeView(
            batch_size=self.num_experts,
            column_groups=2 if self.fused_gate_up else 1,
        ).resolve(storage_shape)


@dataclass(frozen=True, slots=True)
class ExpertDownComputeView:
    """Interpret expert down projections as one Muon matrix per expert.

    Args:
        num_experts: Experts in this parameter after expert parallelism.
    """

    num_experts: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_experts, bool)
            or not isinstance(self.num_experts, int)
            or self.num_experts <= 0
        ):
            raise ValueError("num_experts must be a positive integer")

    def resolve(self, storage_shape: torch.Size) -> ResolvedBatchedMatrixView:
        return BatchedMatrixComputeView(self.num_experts).resolve(storage_shape)


@dataclass(frozen=True, kw_only=True, slots=True)
class MuonParameterSpec:
    """A model parameter's complete-matrix or matrix-batch Muon semantics.

    Owned treats the whole parameter as one matrix. Shard(0) requires a view
    declaring independent matrices. The backend assigns complete matrices to
    compute ranks for both cases, independently of persistent FSDP sharding.
    """

    compute_placement: Owned | Shard
    compute_view: MatrixComputeView | None = None

    def __post_init__(self) -> None:
        if type(self.compute_placement) not in (Owned, Shard):
            raise TypeError("compute_placement must be Owned or Shard")
        if type(self.compute_placement) is Shard and self.compute_placement.dim != 0:
            raise ValueError("Muon sharded compute only supports Shard(0)")
        if type(self.compute_placement) is Shard and self.compute_view is None:
            raise ValueError("Shard(0) Muon compute requires a compute view")
        if type(self.compute_placement) is Owned and self.compute_view is not None:
            raise ValueError("Owned Muon compute does not support a batched view")
        if self.compute_view is not None and not isinstance(
            self.compute_view, MatrixComputeView
        ):
            raise TypeError("compute_view must implement MatrixComputeView")


@dataclass(frozen=True, slots=True)
class ExcludeFromMuon:
    """Keep a parameter out of Muon and leave it to the fallback optimizer."""


# A parameter override is either a Muon matrix layout or an explicit exclusion.
# Providing a layout is itself the statement that the parameter uses Muon, so
# these two states are exhaustive and cannot contradict each other.
MuonParameterOverride = MuonParameterSpec | ExcludeFromMuon


@runtime_checkable
class MuonSpecProvider(Protocol):
    """Module contract for declaring the Muon matrix layouts it owns.

    A module only knows how its own storage is laid out; whether the parameter
    participates in Muon at all is a model-level policy. Keys are parameter
    names local to the module, such as ``"weight1"``.
    """

    def get_muon_parameter_specs(self) -> Mapping[str, MuonParameterSpec]: ...


def collect_module_muon_specs(model: nn.Module) -> dict[str, MuonParameterSpec]:
    """Collect module-declared Muon matrix layouts from the prepared model."""
    specs: dict[str, MuonParameterSpec] = {}
    for module_fqn, module in model.named_modules():
        # Check the class so wrappers forwarding attribute access do not invoke
        # the wrapped module's provider a second time.
        if not issubclass(type(module), MuonSpecProvider):
            continue
        for local_name, spec in module.get_muon_parameter_specs().items():
            # The root module reports an empty FQN, so it owns unprefixed names.
            fqn = f"{module_fqn}.{local_name}" if module_fqn else local_name
            specs[fqn] = spec
    return specs


@dataclass(frozen=True, slots=True)
class MuonRecipe(OptimizerRecipe):
    """A model's Muon parameter overrides, keyed by model FQN patterns.

    Patterns use the same ``*`` and ``**`` syntax as parallel plans. They are
    written against the model's module tree and never account for wrappers in
    the prepared model.

    A recipe only carries what differs from the weaker tiers the engine already
    applies: the default selection rule (two-dimensional non-embedding weights
    that are not PEFT adapters use Muon with ``Owned()`` compute) and the layout
    each module declares through :class:`MuonSpecProvider`.

    ``qk_clip_specs`` declares attention projection layouts separately from
    Muon's matrix views. The engine binds these only when QK clipping is enabled.
    """

    parameter_overrides: Mapping[str, MuonParameterOverride]
    qk_clip_specs: Mapping[str, QKClipSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_overrides, Mapping):
            raise TypeError("MuonRecipe parameter_overrides must be a mapping")
        overrides: dict[str, MuonParameterOverride] = {}
        for pattern, override in self.parameter_overrides.items():
            if not isinstance(pattern, str):
                raise TypeError("MuonRecipe parameter_overrides keys must be strings")
            if not isinstance(override, (MuonParameterSpec, ExcludeFromMuon)):
                raise TypeError(
                    f"MuonRecipe override for {pattern!r} must be a MuonParameterSpec "
                    "or ExcludeFromMuon"
                )
            overrides[pattern] = override
        object.__setattr__(self, "parameter_overrides", MappingProxyType(overrides))
        if not isinstance(self.qk_clip_specs, Mapping) or not all(
            isinstance(pattern, str) and isinstance(spec, QKClipSpec)
            for pattern, spec in self.qk_clip_specs.items()
        ):
            raise TypeError("MuonRecipe qk_clip_specs must map module patterns to QKClipSpec")
        object.__setattr__(self, "qk_clip_specs", MappingProxyType(dict(self.qk_clip_specs)))

    def match(self, fqn: str) -> MuonParameterOverride | None:
        """Return the first override matching a modeling parameter FQN."""
        for pattern, override in self.parameter_overrides.items():
            if match_fqn(pattern, fqn):
                return override
        return None


class MuonPlan(OptimizerPlan):
    """Resolved backend-independent construction plan for distributed Muon."""

    __slots__ = (
        "_param_groups",
        "_param_groups_by_optimizer_name",
        "parameter_specs",
        "buckets",
        "muon_fqns",
        "adamw_no_decay_fqns",
        "adamw_decay_fqns",
        "muon_parameter_bytes",
        "muon_kwargs",
        "adamw_kwargs",
        "overlap_comm_compute",
        "qk_clip_plan",
    )

    name: ClassVar[str] = "muon"
    parameter_specs: Mapping[str, MuonParameterSpec]
    buckets: tuple[MuonBucket, ...]
    muon_fqns: tuple[str, ...]
    adamw_no_decay_fqns: tuple[str, ...]
    adamw_decay_fqns: tuple[str, ...]
    muon_parameter_bytes: int
    muon_kwargs: Mapping[str, Any]
    adamw_kwargs: Mapping[str, Any]
    overlap_comm_compute: bool
    qk_clip_plan: QKClipPlan | None

    def __init__(
        self,
        model: nn.Module,
        *,
        recipe: MuonRecipe,
        config: MuonOptimizerConfig,
        model_fqn_map: ModelFQNMap | None = None,
    ) -> None:
        self._validate(model, recipe, config)
        if model_fqn_map is None:
            model_fqn_map = build_model_fqn_map(model)
        if not isinstance(model_fqn_map, ModelFQNMap):
            raise TypeError("MuonPlan model_fqn_map must be a ModelFQNMap")
        self.qk_clip_plan = None
        if config.qk_clip is not None:
            self.qk_clip_plan = QKClipPlan(
                model,
                specs=recipe.qk_clip_specs,
                config=config.qk_clip,
                model_fqn_map=model_fqn_map,
            )

        named_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        parameter_specs = self._resolve_parameter_specs(
            model,
            recipe,
            model_fqn_map,
        )
        muon_fqns = tuple(parameter_specs)
        fallback = tuple(
            (fqn, parameter)
            for fqn, parameter in named_parameters.items()
            if fqn not in parameter_specs
        )
        adamw_no_decay_fqns = tuple(
            fqn
            for fqn, parameter in fallback
            if fqn.endswith(".bias") or parameter.ndim == 1
        )
        adamw_decay_fqns = tuple(
            fqn
            for fqn, parameter in fallback
            if not (fqn.endswith(".bias") or parameter.ndim == 1)
        )
        parameter_num_bytes = {
            fqn: named_parameters[fqn].numel() * named_parameters[fqn].element_size()
            for fqn in muon_fqns
        }
        bucket_policy = config.bucket
        buckets = _build_muon_buckets(
            muon_fqns,
            strategy=bucket_policy.strategy,
            bucket_cap_mb=bucket_policy.bucket_cap_mb,
            parameter_num_bytes=parameter_num_bytes,
        )
        muon_kwargs = {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "momentum": config.momentum,
            "nesterov": config.nesterov,
            "ns_coefficients": config.ns_coefficients,
            "eps": config.eps,
            "ns_steps": config.ns_steps,
            "adjust_lr_fn": config.adjust_lr_fn,
        }
        # The fallback AdamW rate is set independently of the Muon rate when a
        # framework owns a scalar-optimizer
        adamw_lr = config.lr if config.adamw_lr is None else config.adamw_lr
        adamw_kwargs = {
            "lr": adamw_lr,
            "betas": config.adamw_betas,
            "eps": config.adamw_eps,
            "weight_decay": config.weight_decay,
        }
        if config.adamw_fused is not None:
            adamw_kwargs["fused"] = config.adamw_fused

        self.parameter_specs = MappingProxyType(parameter_specs)
        self.buckets = buckets
        self.muon_fqns = muon_fqns
        self.adamw_no_decay_fqns = adamw_no_decay_fqns
        self.adamw_decay_fqns = adamw_decay_fqns
        self.muon_parameter_bytes = sum(parameter_num_bytes.values())
        self.muon_kwargs = MappingProxyType(muon_kwargs)
        self.adamw_kwargs = MappingProxyType(adamw_kwargs)
        self.overlap_comm_compute = bool(config.overlap_comm_compute)
        self._param_groups = self._prepare_param_groups(model)
        self._validate_param_groups(model, self._param_groups)
        self._param_groups_by_optimizer_name = (
            self._group_param_groups_by_optimizer_name(self._param_groups)
        )

    def get_param_groups(self) -> tuple[ParamGroup, ...]:
        return self._param_groups

    def get_param_groups_by_optimizer_name(
        self,
    ) -> dict[str, tuple[ParamGroup, ...]]:
        return self._param_groups_by_optimizer_name

    @staticmethod
    def _resolve_parameter_specs(
        model: nn.Module,
        recipe: MuonRecipe,
        model_fqn_map: ModelFQNMap,
    ) -> dict[str, MuonParameterSpec]:
        """Merge the default rule, module-declared layouts, and model overrides.

        Precedence runs from weakest to strongest: the engine's default
        selection rule, then the layout each module declares for its own fused
        storage, then the model's recipe.
        """
        from transformers import PreTrainedModel

        embedding_param_ids: set[int] = set()
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                embedding_param_ids.update(id(param) for param in module.parameters())
            if isinstance(module, PreTrainedModel):
                output_embeddings = module.get_output_embeddings()
                if output_embeddings is not None:
                    embedding_param_ids.update(
                        id(param) for param in output_embeddings.parameters()
                    )

        module_specs = collect_module_muon_specs(model)
        model_fqns = tuple(model_fqn_map.parameters)
        unmatched_patterns = [
            pattern
            for pattern in recipe.parameter_overrides
            if not any(match_fqn(pattern, fqn) for fqn in model_fqns)
        ]
        if unmatched_patterns:
            # TODO: Raise ValueError when unmatched recipe patterns become errors.
            logger.warning_rank0(
                "MuonRecipe patterns matched no model parameter FQNs: %s",
                unmatched_patterns,
            )

        recipe_overrides: dict[str, MuonParameterOverride] = {}
        for model_fqn, prepared_fqn in model_fqn_map.parameters.items():
            override = recipe.match(model_fqn)
            if override is not None:
                recipe_overrides[prepared_fqn] = override

        model_parameter_fqns = frozenset(model_fqn_map.parameters.values())
        prepared_fqns = set()
        parameter_specs: dict[str, MuonParameterSpec] = {}
        full_matrix = MuonParameterSpec(compute_placement=Owned())
        for fqn, parameter in model.named_parameters():
            prepared_fqns.add(fqn)
            # Frozen parameters belong to no optimizer, so a layout declared for
            # them is simply inert. This is the normal case under LoRA, where a
            # module still describes base weights that this run does not train.
            if not parameter.requires_grad:
                continue
            override = recipe_overrides.get(fqn)
            if override is None:
                override = module_specs.get(fqn)
            if isinstance(override, ExcludeFromMuon):
                continue
            if override is None:
                # The default rule applies only to modeling parameters. Adapter
                # parameters have no modeling FQN and remain on AdamW unless a
                # module explicitly declares a Muon layout for them.
                if (
                    fqn not in model_parameter_fqns
                    or parameter.ndim != 2
                    or id(parameter) in embedding_param_ids
                ):
                    continue
                spec = full_matrix
            else:
                spec = override
            if parameter.ndim != 2:
                raise ValueError(
                    f"Muon parameter {fqn!r} must be two-dimensional, got "
                    f"{tuple(parameter.shape)}"
                )
            if spec.compute_view is not None:
                spec.compute_view.resolve(torch.Size(parameter.shape))
            parameter_specs[fqn] = spec

        # Module-provided specs are exact prepared FQNs. Recipe keys are
        # patterns, so unmatched keys do not produce unknown-FQN errors.
        unknown_fqns = sorted(set(module_specs) - prepared_fqns)
        if unknown_fqns:
            raise ValueError(
                f"Muon module specs name unknown parameters: "
                f"{unknown_fqns[:8]}"
            )
        if not parameter_specs:
            # A legitimate configuration, for example LoRA where every trainable
            # parameter is an adapter matrix. Warn because the run then behaves
            # exactly like plain AdamW despite requesting Muon.
            logger.warning_rank0(
                "Muon selected no parameters: the model recipe excluded every "
                "candidate or no trainable two-dimensional weights remain. "
                "This run falls back to AdamW for all parameters."
            )
        return parameter_specs

    def _prepare_param_groups(
        self,
        model: nn.Module,
    ) -> tuple[ParamGroup, ...]:
        param_groups = []
        if self.muon_fqns:
            param_groups.append(
                ParamGroup(
                    model,
                    optimizer_name="Muon",
                    param_names=self.muon_fqns,
                    optimizer_kwargs=self.muon_kwargs,
                )
            )
        if self.adamw_no_decay_fqns:
            param_groups.append(
                ParamGroup(
                    model,
                    optimizer_name="AdamW",
                    param_names=self.adamw_no_decay_fqns,
                    optimizer_kwargs={
                        **self.adamw_kwargs,
                        "weight_decay": 0.0,
                    },
                )
            )
        if self.adamw_decay_fqns:
            param_groups.append(
                ParamGroup(
                    model,
                    optimizer_name="AdamW",
                    param_names=self.adamw_decay_fqns,
                    optimizer_kwargs=self.adamw_kwargs,
                )
            )
        return tuple(param_groups)

    @staticmethod
    def _validate(
        model: nn.Module,
        recipe: MuonRecipe,
        config: MuonOptimizerConfig,
    ) -> None:
        from ..config import MuonOptimizerConfig

        if not isinstance(model, nn.Module):
            raise TypeError("MuonPlan model must be an nn.Module")
        if config is None:
            raise ValueError("Muon optimizer configuration is missing")
        if not isinstance(config, MuonOptimizerConfig):
            raise TypeError("MuonPlan config must be a MuonOptimizerConfig")

        missing_config = [
            name
            for name, value in (("lr", config.lr), ("weight_decay", config.weight_decay))
            if value is None
        ]
        if missing_config:
            raise ValueError(
                "Muon optimizer requires configured values for "
                + ", ".join(missing_config)
            )

        if recipe is None:
            raise ValueError(
                f"{type(model).__name__} does not provide a Muon optimizer recipe"
            )
        if not isinstance(recipe, MuonRecipe):
            raise TypeError(
                f"{type(model).__name__} returned {type(recipe).__name__} for the "
                "muon optimizer recipe; a MuonRecipe is required"
            )

        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Muon optimizer requires at least one trainable parameter")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"muon_parameters={len(self.muon_fqns)}, "
            f"muon_parameter_bytes={self.muon_parameter_bytes}, "
            f"adamw_decay_parameters={len(self.adamw_decay_fqns)}, "
            f"adamw_no_decay_parameters={len(self.adamw_no_decay_fqns)}, "
            f"buckets={tuple(bucket.name for bucket in self.buckets)!r}, "
            f"muon_kwargs={dict(self.muon_kwargs)!r}, "
            f"adamw_kwargs={dict(self.adamw_kwargs)!r})"
        )


__all__ = [
    "AttentionPerHeadComputeView",
    "BatchedMatrixComputeView",
    "collect_module_muon_specs",
    "ExcludeFromMuon",
    "ExpertDownComputeView",
    "ExpertUpComputeView",
    "MuonBucket",
    "MuonParameterOverride",
    "MuonParameterSpec",
    "MuonPlan",
    "MuonRecipe",
    "MuonSpecProvider",
    "Owned",
]
