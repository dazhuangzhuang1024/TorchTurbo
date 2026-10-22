"""Language-backbone TP styles and local GatedDeltaNet kernel boundaries."""

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    DTensor, Partial, Placement, Replicate, Shard, distribute_tensor,
)
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, parallelize_module

from torchturbo.engine.tensor_parallel.parallelize import (
    PrepareModuleMetadata,
    SharedHeadNormParallel,
    register_tensor_parallel_cache_guard,
)
from ...parallel import FSDPParallelPlan, TensorParallelValidator
from ..modules.moe.moe_layer import MoELayer
from .modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention, Qwen3_5MoeGatedDeltaNet, Qwen3_5MoeSparseMoeBlock,
)


_GDN_METADATA = ("num_k_heads", "num_v_heads", "key_dim", "value_dim", "conv_dim")


def local_gdn_parameters(
    module: Qwen3_5MoeGatedDeltaNet,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read live TP parameters inside the active unsharded FSDP forward.

    FSDP restores each parameter's one-dimensional TP mesh before these local
    kernels. TP admission requires conv1d.bias=None.
    """
    weight = module.conv1d.weight
    if not isinstance(weight, DTensor):
        return weight, module.A_log, module.dt_bias
    mesh = weight.device_mesh
    size, rank = mesh.size(), mesh.get_local_rank()
    widths = (module.key_dim * size, module.key_dim * size, module.value_dim * size)
    weight = weight.to_local(grad_placements=[Partial()])
    segments = weight.split(widths, dim=0)
    local_weight = torch.cat([segment.chunk(size, dim=0)[rank] for segment in segments])
    return local_weight, module.A_log.to_local(), module.dt_bias.to_local()


def gated_norm(
    module: Qwen3_5MoeGatedDeltaNet, core: torch.Tensor, z: torch.Tensor,
) -> torch.Tensor:
    """Call the selected norm inside the active unsharded FSDP forward."""
    if not isinstance(module.norm, nn.Module) or not isinstance(module.norm.weight, DTensor):
        return module.norm(core, z)
    weight = module.norm.weight.to_local(grad_placements=[Partial()])
    return torch.func.functional_call(module.norm, {"weight": weight}, (core, z))


def _require_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Qwen3.5 TP {name} shape {tuple(tensor.shape)} must be {expected}.")


def _validate_attention(module: Qwen3_5MoeAttention, tp_size: int, sp_size: int) -> None:
    hidden, dim = module.config.hidden_size, module.head_dim
    q_width, k_width = module.q_proj.out_features, module.k_proj.out_features
    if q_width <= 0 or q_width % (2 * dim * tp_size):
        raise ValueError("Qwen3.5 TP q_proj must partition whole query/gate head pairs.")
    for name, projection in (("k_proj", module.k_proj), ("v_proj", module.v_proj)):
        if projection.out_features <= 0 or projection.out_features % (dim * tp_size):
            raise ValueError(f"Qwen3.5 TP {name} must partition whole KV heads.")
    q_heads, k_heads = q_width // (2 * dim), k_width // dim
    if q_heads % k_heads or q_heads // k_heads != module.num_key_value_groups:
        raise ValueError("Qwen3.5 TP requires the actual query/KV head ratio to match GQA.")
    for name, projection, width in (
        ("q_proj", module.q_proj, module.config.num_attention_heads * dim * 2),
        ("k_proj", module.k_proj, module.config.num_key_value_heads * dim),
        ("v_proj", module.v_proj, k_width),
    ):
        _require_shape(name, projection.weight, (width, hidden))
    _require_shape("o_proj", module.o_proj.weight, (hidden, q_width // 2))
    _require_shape("q_norm", module.q_norm.weight, (dim,))
    _require_shape("k_norm", module.k_norm.weight, (dim,))
    if module.attention_dropout != 0:
        raise ValueError("Qwen3.5 TP requires attention_dropout=0.")
    local_k = k_heads // tp_size
    repeated_k = local_k * max(1, sp_size // local_k)
    if repeated_k % sp_size:
        raise ValueError("Qwen3.5 TP-local KV heads are incompatible with SP.")


def _validate_gdn(module: Qwen3_5MoeGatedDeltaNet, tp_size: int, sp_size: int) -> None:
    PrepareModuleMetadata(divide=_GDN_METADATA).validate(module, tp_size)
    nk, nv = module.num_k_heads, module.num_v_heads
    if nk < 1 or nv < 1 or nv % nk:
        raise ValueError("Qwen3.5 TP value heads must be a positive multiple of key heads.")
    if nk % tp_size or nv % tp_size:
        raise ValueError("Qwen3.5 TP num_k_heads and num_v_heads must be divisible by TP size.")
    hidden = module.hidden_size
    if hidden % tp_size:
        raise ValueError("Qwen3.5 TP hidden_size must be divisible by TP size.")
    key, value = nk * module.head_k_dim, nv * module.head_v_dim
    conv = 2 * key + value
    if (module.key_dim, module.value_dim, module.conv_dim) != (key, value, conv):
        raise ValueError("Qwen3.5 TP GDN key/value/conv metadata must match complete heads.")
    for name, projection, width in (
        ("in_proj_qkv", module.in_proj_qkv, conv),
        ("in_proj_z", module.in_proj_z, value),
        ("in_proj_a", module.in_proj_a, nv),
        ("in_proj_b", module.in_proj_b, nv),
    ):
        _require_shape(name, projection.weight, (width, hidden))
        if projection.bias is not None:
            raise ValueError(f"Qwen3.5 TP {name} requires bias=None.")
    _require_shape("out_proj", module.out_proj.weight, (hidden, value))
    _require_shape("conv1d", module.conv1d.weight, (conv, 1, module.conv_kernel_size))
    _require_shape("A_log", module.A_log, (nv,))
    _require_shape("dt_bias", module.dt_bias, (nv,))
    _require_shape("norm", module.norm.weight, (module.head_v_dim,))
    if module.conv1d.bias is not None:
        raise ValueError("Qwen3.5 TP conv1d requires bias=None.")
    if module.activation != "silu":
        raise ValueError("Qwen3.5 TP GDN requires activation='silu'.")
    local_k, local_v = nk // tp_size, nv // tp_size
    super_heads = local_k * max(1, sp_size // local_k)
    if super_heads % sp_size or local_v % super_heads:
        raise ValueError("Qwen3.5 TP-local GDN heads are incompatible with SP.")


def _distribute_parameter(
    module: nn.Module, name: str, mesh: DeviceMesh, placement: Placement,
) -> None:
    parameter = module.get_parameter(name)
    distributed = distribute_tensor(parameter, mesh, [placement])
    module.register_parameter(name, nn.Parameter(distributed, requires_grad=parameter.requires_grad))


def _parallelize_gdn(module: Qwen3_5MoeGatedDeltaNet, mesh: DeviceMesh) -> None:
    widths = (module.key_dim, module.key_dim, module.value_dim)
    parallelize_module(module.in_proj_qkv, mesh, RowwiseParallel(
        input_layouts=Replicate(), output_layouts=Replicate(), use_local_output=False,
    ))

    def local_qkv(projection: nn.Module, inputs: tuple, output: DTensor) -> torch.Tensor:
        segments = output.split(widths, dim=-1)
        return torch.cat([
            segment.redistribute(placements=[Shard(-1)]).to_local()
            for segment in segments
        ], dim=-1)

    module.in_proj_qkv.register_forward_hook(local_qkv)
    for projection in (module.in_proj_z, module.in_proj_a, module.in_proj_b):
        parallelize_module(projection, mesh, ColwiseParallel())
    parallelize_module(module.out_proj, mesh, RowwiseParallel(input_layouts=Shard(-1)))
    _distribute_parameter(module.conv1d, "weight", mesh, Replicate())
    _distribute_parameter(module.norm, "weight", mesh, Replicate())
    for name in ("A_log", "dt_bias"):
        _distribute_parameter(module, name, mesh, Shard(0))
    metadata = PrepareModuleMetadata(divide=_GDN_METADATA)
    parallelize_module(module, mesh, metadata)


class Qwen3_5MoeTensorParallelValidator(TensorParallelValidator):
    """Validate actual language decoders on original and prepared trees."""

    backbone: str

    def __init__(self, backbone: str) -> None:
        self.backbone = backbone

    def validate(self, model: nn.Module, parallel_plan: FSDPParallelPlan) -> None:
        tp_size, sp_size = parallel_plan.tp_size, parallel_plan.sp_size
        if model.mtp is not None:
            raise ValueError("Qwen3.5 TP does not support an active MTP module.")
        backbone = model.get_submodule(self.backbone)
        for index, layer in enumerate(backbone.layers):
            if not isinstance(layer.mlp, (Qwen3_5MoeSparseMoeBlock, MoELayer)):
                raise ValueError(f"Qwen3.5 TP requires routed MLPs; layer {index} is unsupported.")
            if layer.layer_type == "full_attention" and isinstance(layer.self_attn, Qwen3_5MoeAttention):
                _validate_attention(layer.self_attn, tp_size, sp_size)
            elif layer.layer_type == "linear_attention" and isinstance(layer.linear_attn, Qwen3_5MoeGatedDeltaNet):
                _validate_gdn(layer.linear_attn, tp_size, sp_size)
            else:
                raise ValueError(f"Qwen3.5 TP does not support layer type {layer.layer_type!r}.")
            shared = layer.mlp.shared_expert
            features = shared.gate_proj.out_features
            if features % tp_size:
                raise ValueError("Qwen3.5 TP shared_expert width must be divisible by TP size.")
            for name, projection in (("gate_proj", shared.gate_proj), ("up_proj", shared.up_proj)):
                _require_shape(f"shared_expert.{name}", projection.weight, (features, layer.hidden_size))
            _require_shape("shared_expert.down_proj", shared.down_proj.weight, (layer.hidden_size, features))
            if isinstance(layer.mlp, Qwen3_5MoeSparseMoeBlock):
                if layer.mlp.experts.intermediate_dim % tp_size:
                    raise ValueError("Qwen3.5 ETP expert width must be divisible by ETP size.")


class Qwen3_5MoeTensorParallel(ParallelStyle):
    """Partition language decoder heads/features and retain complete tokens."""

    backbone: str

    def __init__(self, backbone: str) -> None:
        super().__init__()
        self.backbone = backbone

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        backbone = module.get_submodule(self.backbone)
        for layer in backbone.layers:
            if layer.layer_type == "linear_attention":
                _parallelize_gdn(layer.linear_attn, device_mesh)
            else:
                attention = layer.self_attn
                for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
                    parallelize_module(projection, device_mesh, ColwiseParallel())
                parallelize_module(attention.o_proj, device_mesh, RowwiseParallel(input_layouts=Shard(-1)))
                for norm in (attention.q_norm, attention.k_norm):
                    parallelize_module(norm, device_mesh, SharedHeadNormParallel(head_axis=2))
            shared = layer.mlp.shared_expert
            for projection in (shared.gate_proj, shared.up_proj):
                parallelize_module(projection, device_mesh, ColwiseParallel())
            parallelize_module(shared.down_proj, device_mesh, RowwiseParallel(input_layouts=Shard(-1)))
        register_tensor_parallel_cache_guard(module)
        register_tensor_parallel_cache_guard(backbone)
        return module
