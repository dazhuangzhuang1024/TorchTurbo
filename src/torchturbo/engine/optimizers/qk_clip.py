"""Model semantics and parameter binding for QK clipping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from ..model_fqn import ModelFQNMap
from ...utils.fqn import match_fqn

if TYPE_CHECKING:
    from ..config import QKClipConfig


class QKClipSpec(ABC):
    """Declare attention heads and projections participating in QK clipping.

    Concrete specs describe the projection layout. Backends implement the
    corresponding rescaling rule for the spec types they support.
    """

    __slots__ = ()

    num_heads: int

    @property
    @abstractmethod
    def projection_rows(self) -> Mapping[str, int]:
        """Map projection names relative to attention to expected output rows."""
        ...


@dataclass(frozen=True, slots=True)
class MLAQKClipSpec(QKClipSpec):
    """Describe the final query and key/value projections of MLA.

    Projection names are relative to the attention module. Each query head has
    contiguous NoPE then RoPE rows; each key/value head has K then V rows.
    The shared RoPE key projection is deliberately excluded.
    """

    query_projection: str
    key_value_projection: str
    num_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    def __post_init__(self) -> None:
        dimensions = (
            self.num_heads,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("MLA QK clip head counts and dimensions must be positive integers")
        if not self.query_projection or not self.key_value_projection:
            raise ValueError("MLA QK clip requires query and key/value projection names")
        if self.query_projection == self.key_value_projection:
            raise ValueError("MLA QK clip requires distinct projection names")

    @property
    def projection_rows(self) -> Mapping[str, int]:
        return {
            self.query_projection: self.num_heads * (
                self.qk_nope_head_dim + self.qk_rope_head_dim
            ),
            self.key_value_projection: self.num_heads * (
                self.qk_nope_head_dim + self.v_head_dim
            ),
        }


@dataclass(frozen=True, slots=True)
class QKClipLayer:
    """An attention module and its bound, persistent projection parameters."""

    module: nn.Module
    weights: Mapping[str, nn.Parameter]
    spec: QKClipSpec


@dataclass(frozen=True, slots=True, init=False)
class QKClipPlan:
    """Resolve model-authored attention patterns before backend installation."""

    config: QKClipConfig
    layers: tuple[QKClipLayer, ...]

    def __init__(
        self,
        model: nn.Module,
        *,
        specs: Mapping[str, QKClipSpec],
        config: QKClipConfig,
        model_fqn_map: ModelFQNMap,
    ) -> None:
        if not specs:
            raise ValueError("QK clipping requires an explicit QK clip recipe for this model")
        layers = []
        seen_modules: set[int] = set()
        seen_parameters: set[int] = set()
        for pattern, spec in specs.items():
            projection_rows = spec.projection_rows
            if not projection_rows:
                raise ValueError("QK clip specs must declare at least one projection")
            matches = [
                (name, prepared_name)
                for name, prepared_name in model_fqn_map.modules.items()
                if match_fqn(pattern, name)
            ]
            if not matches:
                raise ValueError(f"QK clip pattern {pattern!r} matches no attention module")
            for name, prepared_name in matches:
                module = model.get_submodule(prepared_name)
                while isinstance(module, CheckpointWrapper):
                    module = module._checkpoint_wrapped_module
                if id(module) in seen_modules:
                    raise ValueError(f"QK clip recipes overlap at attention {name!r}")
                seen_modules.add(id(module))
                weights = {
                    projection_name: self._projection(
                        model, module, name, projection_name, model_fqn_map
                    )
                    for projection_name in projection_rows
                }
                if not any(weight.requires_grad for weight in weights.values()):
                    continue
                if not all(weight.requires_grad for weight in weights.values()):
                    raise ValueError(
                        f"QK clip attention {name!r} requires all final projections "
                        "to be trainable"
                    )
                for projection_name, parameter in weights.items():
                    if parameter.shape[0] != projection_rows[projection_name]:
                        raise ValueError(
                            f"QK clip dimensions do not match attention {name!r} "
                            f"projection {projection_name!r}"
                        )
                    if id(parameter) in seen_parameters:
                        raise ValueError("QK clip does not support shared projection weights")
                    seen_parameters.add(id(parameter))
                layers.append(QKClipLayer(module, MappingProxyType(weights), spec))
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "layers", tuple(layers))

    @staticmethod
    def _projection(
        model: nn.Module,
        attention: nn.Module,
        attention_fqn: str,
        projection_name: str,
        model_fqn_map: ModelFQNMap,
    ) -> nn.Parameter:
        projection = attention.get_submodule(projection_name)
        while isinstance(projection, CheckpointWrapper):
            projection = projection._checkpoint_wrapped_module
        if not isinstance(projection, nn.Linear) or projection.bias is not None:
            raise ValueError("QK clipping requires bias-free Linear projections without adapters")
        weight_fqn = f"{attention_fqn}.{projection_name}.weight"
        weight = model.get_parameter(model_fqn_map.parameters[weight_fqn])
        if weight is not projection.weight:
            raise ValueError(
                f"QK clip projection {weight_fqn!r} must identify its active base weight"
            )
        return weight
