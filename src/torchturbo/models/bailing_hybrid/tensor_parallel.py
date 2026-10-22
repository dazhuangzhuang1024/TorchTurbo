"""Native language-backbone tensor parallelism for Bailing model families."""

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, parallelize_module

from torchturbo.engine.tensor_parallel.parallelize import (
    register_tensor_parallel_cache_guard,
)
from ..modules.moe.experts import GroupedMLP
from ...parallel import FSDPParallelPlan, TensorParallelValidator
from ..modules.moe.moe_layer import MoELayer


def _family_types(backbone: nn.Module) -> tuple[type[nn.Module], ...]:
    from .modeling_bailing_moe_v3 import (
        BailingMoeV3Model, BailingMoeV3DecoderLayer, BailingMoeV3MultiLatentAttention,
        BailingMoeV3KimiDeltaAttention, BailingMoeV3MLP, BailingMoeV3SparseMoeBlock,
    )
    if isinstance(backbone, BailingMoeV3Model):
        return (BailingMoeV3DecoderLayer, BailingMoeV3MultiLatentAttention,
                BailingMoeV3KimiDeltaAttention, BailingMoeV3MLP, BailingMoeV3SparseMoeBlock)
    from ..bailing_moe_v3_vl.modeling_bailing_moe_v3_vl import (
        BailingMoeV3Model, BailingMoeV3DecoderLayer, BailingMoeV3MultiLatentAttention,
        BailingMoeV3KimiDeltaAttention, BailingMoeV3MLP, BailingMoeV3SparseMoeBlock,
    )
    if not isinstance(backbone, BailingMoeV3Model):
        raise ValueError("Bailing TP requires an actual Bailing language backbone.")
    return (BailingMoeV3DecoderLayer, BailingMoeV3MultiLatentAttention,
            BailingMoeV3KimiDeltaAttention, BailingMoeV3MLP, BailingMoeV3SparseMoeBlock)


def _shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Bailing TP {name} shape {tuple(tensor.shape)} must be {expected}.")


def _linear(name: str, projection: nn.Linear, inputs: int, outputs: int, bias: bool = False) -> None:
    _shape(name, projection.weight, (outputs, inputs))
    if bias:
        if projection.bias is None:
            raise ValueError(f"Bailing TP {name} requires its configured bias.")
        _shape(name + ".bias", projection.bias, (outputs,))
    elif projection.bias is not None:
        raise ValueError(f"Bailing TP {name} requires bias=None.")


def _mlp(name: str, module: nn.Module, hidden: int, features: int, size: int) -> None:
    if features < 1 or features % size:
        raise ValueError(f"Bailing TP {name} features must be divisible by TP/ETP size.")
    for suffix in ("gate_proj", "up_proj"):
        _linear(name + "." + suffix, module.get_submodule(suffix), hidden, features)
    _linear(name + ".down_proj", module.down_proj, features, hidden)


def _mla(module: nn.Module, size: int, sp: int) -> None:
    config = module.config
    hidden, heads = config.hidden_size, config.num_attention_heads
    if heads < 1 or heads % (size * sp) or module.num_heads != heads:
        raise ValueError("Bailing TP physical attention heads must be divisible by TP*SP.")
    p, r, v, c = config.qk_nope_head_dim, config.qk_rope_head_dim, config.v_head_dim, config.kv_lora_rank
    for actual, expected in ((module.qk_nope_head_dim, p), (module.qk_rope_head_dim, r),
                             (module.v_head_dim, v), (module.kv_lora_rank, c),
                             (module.qk_head_dim, p + r), (module.q_lora_rank, config.q_lora_rank)):
        if actual != expected:
            raise ValueError("Bailing TP MLA head/latent metadata must match global config.")
    if module.q_lora_rank is None:
        _linear("q_proj", module.q_proj, hidden, heads * (p + r))
    else:
        rank = module.q_lora_rank
        _linear("q_a_proj", module.q_a_proj, hidden, rank, config.use_qkv_bias)
        _shape("q_a_layernorm", module.q_a_layernorm.weight, (rank,))
        _linear("q_b_proj", module.q_b_proj, rank, heads * (p + r))
    _linear("kv_a_proj_with_mqa", module.kv_a_proj_with_mqa, hidden, c + r, config.use_qkv_bias)
    _shape("kv_a_layernorm", module.kv_a_layernorm.weight, (c,))
    _linear("kv_b_proj", module.kv_b_proj, c, heads * (p + v))
    _linear("dense", module.dense, heads * v, hidden, config.use_qkv_bias)
    gate = module.gated_attention_proj_granularity_type
    if gate not in (None, "head_wise", "element_wise"):
        raise ValueError("Bailing TP MLA gate kind is unsupported.")
    if gate is None:
        if module.g_proj is not None:
            raise ValueError("Bailing TP ungated MLA requires g_proj=None.")
    else:
        _linear("g_proj", module.g_proj, hidden, heads if gate == "head_wise" else heads * v)
    if module.attention_dropout != 0:
        raise ValueError("Bailing TP requires live MLA attention_dropout=0.")


def _kda(module: nn.Module, size: int, sp: int) -> None:
    config = module.config
    hidden, heads, dim = config.hidden_size, config.num_attention_heads, config.head_dim
    if heads < 1 or heads % (size * sp) or module.num_heads != heads or module.num_k_heads != heads:
        raise ValueError("Bailing TP KDA heads must be equal and divisible by TP*SP.")
    if (module.hidden_size, module.head_dim, module.head_k_dim, module.conv_size) != (
        hidden, dim, dim, config.short_conv_kernel_size
    ):
        raise ValueError("Bailing TP KDA metadata must match global head/conv shapes.")
    width = heads * dim
    for name in ("q_proj", "k_proj", "v_proj"):
        _linear(name, module.get_submodule(name), hidden, width)
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        conv = module.get_submodule(name)
        _shape(name, conv.weight, (width, 1, module.conv_size))
        if conv.bias is not None:
            raise ValueError(f"Bailing TP {name} requires bias=None.")
    _shape("A_log", module.A_log, (heads,))
    _shape("dt_bias", module.dt_bias, (width,))
    _shape("o_norm", module.o_norm.weight, (dim,))
    _linear("b_proj", module.b_proj, hidden, heads)
    if module.no_kda_lora != config.no_kda_lora:
        raise ValueError("Bailing TP KDA projection choice must match config.")
    for prefix in ("f", "g"):
        if module.no_kda_lora:
            _linear(prefix + "_proj", module.get_submodule(prefix + "_proj"), hidden, width)
        else:
            _linear(prefix + "_a_proj", module.get_submodule(prefix + "_a_proj"), hidden, dim)
            _linear(prefix + "_b_proj", module.get_submodule(prefix + "_b_proj"), dim, width)
    _linear("o_proj", module.o_proj, width, hidden)


def _distribute_parameter(module: nn.Module, name: str, mesh: DeviceMesh,
                          placement: Shard | Replicate) -> None:
    parameter = module.get_parameter(name)
    module.register_parameter(name, nn.Parameter(
        distribute_tensor(parameter, mesh, [placement]), requires_grad=parameter.requires_grad
    ))


class BailingTensorParallelValidator(TensorParallelValidator):
    """Validate concrete original and prepared Bailing decoder inventories."""

    def validate(self, model: nn.Module, parallel_plan: FSDPParallelPlan) -> None:
        tp_size, sp_size = parallel_plan.tp_size, parallel_plan.sp_size
        backbone = model.model
        decoder, mla, kda, dense, sparse = _family_types(backbone)
        config = backbone.config
        if config.attention_dropout != 0:
            raise ValueError("Bailing TP requires text attention_dropout=0.")
        routed = False
        for layer in backbone.layers:
            if not isinstance(layer, decoder):
                raise ValueError("Bailing TP does not support an active MTP or unsupported decoder.")
            if isinstance(layer.attention, mla) and layer.attention_layer_type == "attention":
                _mla(layer.attention, tp_size, sp_size)
            elif isinstance(layer.attention, kda) and layer.attention_layer_type == "linear_attention":
                _kda(layer.attention, tp_size, sp_size)
            else:
                raise ValueError("Bailing TP attention inventory does not match its decoder type.")
            mlp = layer.mlp
            if isinstance(mlp, dense):
                _mlp("dense", mlp, config.hidden_size, config.intermediate_size, tp_size)
            elif isinstance(mlp, sparse):
                routed = True
                if len(mlp.experts) != config.num_experts:
                    raise ValueError("Bailing TP routed expert inventory must match global config.")
                for expert in mlp.experts:
                    if not isinstance(expert, dense):
                        raise ValueError("Bailing TP requires actual routed expert MLPs.")
                    _mlp("expert", expert, config.hidden_size, config.moe_intermediate_size, tp_size)
                if config.num_shared_experts is not None:
                    _mlp("shared", mlp.shared_experts, config.hidden_size,
                         config.moe_shared_expert_intermediate_size * config.num_shared_experts, tp_size)
            elif isinstance(mlp, MoELayer):
                routed = True
                if not isinstance(mlp.experts, GroupedMLP):
                    raise ValueError("Bailing TP requires prepared GroupedMLP experts.")
                features = mlp.config.moe_ffn_hidden_size
                if (features, mlp.config.num_moe_experts) != (
                    config.moe_intermediate_size, config.num_experts
                ):
                    raise ValueError("Bailing TP prepared expert metadata must match global config.")
                if features < 1 or features % tp_size:
                    raise ValueError("Bailing ETP expert features must be divisible by ETP size.")
                _shape("expert.weight1", mlp.experts.weight1,
                       (mlp.num_local_experts * config.hidden_size, 2 * features))
                _shape("expert.weight2", mlp.experts.weight2,
                       (mlp.num_local_experts * features, config.hidden_size))
                if mlp.shared_experts is not None:
                    _mlp("shared", mlp.shared_experts, config.hidden_size,
                         config.moe_shared_expert_intermediate_size * config.num_shared_experts, tp_size)
            else:
                raise ValueError("Bailing TP requires actual dense or routed MLP inventory.")
        if not routed:
            raise ValueError("Bailing TP requires at least one routed MLP layer.")


class BailingTensorParallel(ParallelStyle):
    """Partition concrete decoder heads/features and install cache guards."""

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        backbone = module.model
        _, _, kda, dense, sparse = _family_types(backbone)
        for layer in backbone.layers:
            attention = layer.attention
            if isinstance(attention, kda):
                projections = [attention.q_proj, attention.k_proj, attention.v_proj,
                               attention.b_proj]
                if attention.no_kda_lora:
                    projections.extend((attention.f_proj, attention.g_proj))
                else:
                    projections.extend((attention.f_b_proj, attention.g_b_proj))
                for projection in projections:
                    parallelize_module(projection, device_mesh, ColwiseParallel())
                parallelize_module(attention.o_proj, device_mesh,
                                   RowwiseParallel(input_layouts=Shard(-1)))
                for conv in (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d):
                    _distribute_parameter(conv, "weight", device_mesh, Shard(0))
                for name in ("A_log", "dt_bias"):
                    _distribute_parameter(attention, name, device_mesh, Shard(0))
                _distribute_parameter(attention.o_norm, "weight", device_mesh, Replicate())
                attention.num_k_heads //= device_mesh.size()
            else:
                query = attention.q_proj if attention.q_lora_rank is None else attention.q_b_proj
                for projection in (query, attention.kv_b_proj):
                    parallelize_module(projection, device_mesh, ColwiseParallel())
                if attention.g_proj is not None:
                    parallelize_module(attention.g_proj, device_mesh, ColwiseParallel())
                parallelize_module(attention.dense, device_mesh, RowwiseParallel(input_layouts=Shard(-1)))
            attention.num_heads //= device_mesh.size()
            mlp = layer.mlp
            shared = None
            if isinstance(mlp, dense):
                shared = mlp
            elif isinstance(mlp, sparse) and backbone.config.num_shared_experts is not None:
                shared = mlp.shared_experts
            elif isinstance(mlp, MoELayer):
                shared = mlp.shared_experts
            if shared is not None:
                for projection in (shared.gate_proj, shared.up_proj):
                    parallelize_module(projection, device_mesh, ColwiseParallel())
                parallelize_module(shared.down_proj, device_mesh, RowwiseParallel(input_layouts=Shard(-1)))
        backbone.config.use_cache = False
        register_tensor_parallel_cache_guard(module)
        register_tensor_parallel_cache_guard(backbone)
        return module
