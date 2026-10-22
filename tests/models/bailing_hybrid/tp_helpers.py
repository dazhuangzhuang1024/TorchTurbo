"""Actual Bailing constructors and independent MLA mathematics for TP tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from typing import Any

import pytest
import torch
from torch.nn import functional as F
from transformers import PretrainedConfig


def make_bailing_config(family: str, **overrides: Any) -> PretrainedConfig:
    values = dict(
        vocab_size=64, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=1, head_dim=4,
        qk_nope_head_dim=4, qk_rope_head_dim=8, v_head_dim=4, kv_lora_rank=8,
        num_experts=8, num_shared_experts=1, moe_intermediate_size=12,
        moe_shared_expert_intermediate_size=12, num_experts_per_tok=2,
        n_group=2, topk_group=1, layer_group_size=2, first_k_dense_replace=1,
        routed_scaling_factor=1.3, pad_token_id=0, eos_token_id=1,
        num_nextn_predict_layers=0, expert_swiglu_limit_list=[0.0, 0.4],
        share_expert_swiglu_limit_list=[0.0, 0.5],
    )
    values.update(overrides)
    if family == "hybrid":
        from torchturbo.models.bailing_hybrid.configuration_bailing_moe_v3 import BailingMoeV3Config
        return BailingMoeV3Config(**values)
    from torchturbo.models.bailing_moe_v3_vl.configuration_bailing_moe_v3_vl import BailingMoeV3VLConfig
    return BailingMoeV3VLConfig(
        text_config=values, mrope_section=[2, 1, 1], image_token_id=60,
        video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=24,
                           num_heads=2, patch_size=2, temporal_patch_size=2,
                           spatial_merge_size=2, num_position_embeddings=16,
                           disable_merger_proj=True, deepstack_visual_indexes=[]),
    )


def make_bailing_model(family: str, **overrides: Any) -> torch.nn.Module:
    from torchturbo.engine.config import EngineConfig
    if family == "hybrid":
        from torchturbo.models.bailing_hybrid.model import TurboBailingMoeV3ForCausalLM
        cls = TurboBailingMoeV3ForCausalLM
    else:
        from torchturbo.models.bailing_moe_v3_vl.model import TurboBailingMoeV3VLForConditionalGeneration
        cls = TurboBailingMoeV3VLForConditionalGeneration
    config = make_bailing_config(family, **overrides)
    with cls.init_context(config, EngineConfig()):
        model = cls(config)
    if next(model.parameters()).device.type != "meta":
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                numbers = torch.arange(parameter.numel()).reshape(parameter.shape).float()
                if name.endswith("dt_bias"):
                    parameter.copy_(torch.linspace(-0.3, 0.3, parameter.numel()))
                elif name.endswith("A_log"):
                    parameter.copy_(torch.linspace(-1.2, -0.4, parameter.numel()))
                elif "norm" in name and parameter.ndim == 1:
                    parameter.copy_(torch.linspace(0.7, 1.3, parameter.numel()))
                else:
                    parameter.copy_(0.15 * torch.sin(numbers * 0.17 + 0.31))
            for name, buffer in model.named_buffers():
                if name.endswith("expert_bias"):
                    buffer.copy_(torch.linspace(-0.05, 0.05, buffer.numel()))
    return model


@contextmanager
def cpu_bailing_leaves(events: list[dict[str, Any]] | None = None) -> Iterator[None]:
    """Guard FA2 construction and scope the five CPU mathematical leaves."""
    from packaging.version import Version
    from transformers import __version__ as transformers_version
    import transformers.modeling_utils as hf_modeling
    from tests.models.modules.attention.tp_helpers import install_cpu_attention_bridge
    from torchturbo.ops import get_or_create_ops
    from torchturbo.runtime.cpu import CpuRuntime

    monkeypatch = pytest.MonkeyPatch()
    try:
        # Scope the class methods patched by constructors and legacy calls.
        for family_module in ("bailing_hybrid.modeling_bailing_moe_v3",
                              "bailing_moe_v3_vl.modeling_bailing_moe_v3_vl"):
            modeling = import_module("torchturbo.models." + family_module)
            for cls in (modeling.BailingMoeV3RMSNorm, modeling.BailingMoeV3KimiDeltaAttention):
                monkeypatch.setattr(cls, "forward", cls.forward)
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        for registry in (ALL_MASK_ATTENTION_FUNCTIONS, ALL_ATTENTION_FUNCTIONS):
            monkeypatch.setattr(type(registry), "_global_mapping", registry._global_mapping.copy())
            monkeypatch.setattr(registry, "_local_mapping", registry._local_mapping.copy())
        dispatcher_module = import_module("torchturbo.ops.ops")
        monkeypatch.setattr(dispatcher_module, "_OPS", dispatcher_module._OPS)
        get_or_create_ops(device_runtime=CpuRuntime())
        if Version(transformers_version) >= Version("5.4.0"):
            monkeypatch.setitem(hf_modeling.FLASH_ATTENTION_COMPATIBILITY_MATRIX[2],
                                "general_availability_check", lambda: True)
        else:
            monkeypatch.setattr(hf_modeling, "is_flash_attn_2_available", lambda: True)
        monkeypatch.setattr(hf_modeling, "lazy_import_flash_attention", lambda *args: None)
        install_cpu_attention_bridge(monkeypatch)
        install_cpu_kda_bridge(monkeypatch, events)
        install_cpu_vision_bridge(monkeypatch, events)
        yield
    finally:
        monkeypatch.undo()


def mla_oracle(module: torch.nn.Module, hidden: torch.Tensor,
               positions: tuple[torch.Tensor, torch.Tensor],
               boundaries: torch.Tensor) -> torch.Tensor:
    """Full-weight expanded-head MLA with independent packed segment attention."""
    config = module.config
    heads, p, r = config.num_attention_heads, module.qk_nope_head_dim, module.qk_rope_head_dim
    def linear(projection, value):
        return F.linear(value, projection.weight, projection.bias)
    def normalize(norm, value):
        normalized = value.float() * (value.float().square().mean(-1, keepdim=True)
                                      + norm.variance_epsilon).rsqrt()
        return norm.weight * normalized.to(value.dtype)
    if module.q_lora_rank is None:
        query = linear(module.q_proj, hidden)
    else:
        query = linear(module.q_b_proj, normalize(module.q_a_layernorm,
                                                 linear(module.q_a_proj, hidden)))
    query = query.reshape(*hidden.shape[:2], heads, p + r).transpose(1, 2)
    q_pass, q_rot = query.split((p, r), -1)
    latent, k_rot = linear(module.kv_a_proj_with_mqa, hidden).split((module.kv_lora_rank, r), -1)
    expanded = linear(module.kv_b_proj, normalize(module.kv_a_layernorm, latent))
    expanded = expanded.reshape(*hidden.shape[:2], heads, p + module.v_head_dim).transpose(1, 2)
    key, value = expanded.split((p, module.v_head_dim), -1)
    k_rot = k_rot.unsqueeze(1)
    if config.rope_interleave:
        q_rot = q_rot.reshape(*q_rot.shape[:-1], r // 2, 2).transpose(-1, -2).reshape_as(q_rot)
        k_rot = k_rot.reshape(*k_rot.shape[:-1], r // 2, 2).transpose(-1, -2).reshape_as(k_rot)
    cos, sin = (position.unsqueeze(1) for position in positions)
    def rotate(tensor):
        left, right = tensor.chunk(2, -1)
        return torch.cat((-right, left), -1)
    q_rot = q_rot * cos + rotate(q_rot) * sin
    k_rot = k_rot * cos + rotate(k_rot) * sin
    query = torch.cat((q_pass, q_rot), -1)
    key = torch.cat((key, k_rot.expand_as(q_rot)), -1)
    outputs = []
    for start, end in zip(boundaries.tolist(), boundaries.tolist()[1:]):
        outputs.append(F.scaled_dot_product_attention(
            query[:, :, start:end], key[:, :, start:end], value[:, :, start:end],
            is_causal=True, scale=module.scaling))
    output = torch.cat(outputs, 2).transpose(1, 2)
    if module.g_proj is not None:
        gate = linear(module.g_proj, hidden).float().sigmoid().to(hidden.dtype)
        if module.gated_attention_proj_granularity_type == "head_wise":
            output = output * gate.unsqueeze(-1)
        else:
            output = output * gate.reshape_as(output)
    return linear(module.dense, output.flatten(2))


def mlp_oracle(module: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    gate = module.act_fn(F.linear(hidden, module.gate_proj.weight))
    up = F.linear(hidden, module.up_proj.weight)
    if module.swiglu_limit is not None and module.swiglu_limit > 0:
        gate = gate.clamp(max=module.swiglu_limit)
        up = up.clamp(-module.swiglu_limit, module.swiglu_limit)
    return F.linear(gate * up, module.down_proj.weight)


def cpu_causal_conv1d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, *,
    activation: str | None = None, initial_state: torch.Tensor | None = None,
    output_final_state: bool = False, packed_params: Any = None, impl: str | None = None,
) -> tuple[torch.Tensor, None]:
    """Bridge packed causal convolution through per-segment depthwise CPU conv."""
    from torch.distributed.tensor import DTensor
    assert not any(isinstance(t, DTensor) for t in (x, weight, bias))
    assert initial_state is None and not output_final_state
    assert x.ndim == 3 and x.shape[0] == 1
    assert weight.ndim == 2 and weight.shape[0] == x.shape[-1]
    assert activation in (None, "silu", "swish")
    offsets = packed_params.cu_seqlens_list if packed_params is not None else [0, x.shape[1]]
    segments = []
    for start, end in zip(offsets, offsets[1:]):
        segment = x[:, start:end].transpose(1, 2)
        convolved = F.conv1d(F.pad(segment, (weight.shape[-1] - 1, 0)),
                             weight.unsqueeze(1), bias, groups=x.shape[-1])
        if activation is not None:
            convolved = F.silu(convolved)
        segments.append(convolved.transpose(1, 2))
    return torch.cat(segments, 1), None


def cpu_chunk_kda(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, *, A_log: torch.Tensor, dt_bias: torch.Tensor,
    initial_state: torch.Tensor | None = None, output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False, use_gate_in_kernel: bool = False,
    safe_gate: bool = False, lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Bounded equal-head FP32 recurrence from pinned FLA source mathematics."""
    from torch.distributed.tensor import DTensor
    assert not any(isinstance(t, DTensor) for t in (q, k, v, g, beta, A_log, dt_bias))
    if q.ndim != 4 or q.shape != k.shape:
        raise ValueError("KDA requires matching Q/K shapes")
    if v.shape != q.shape or g.shape != q.shape or beta.shape != q.shape[:-1]:
        raise ValueError("Bailing KDA requires equal heads/widths and matching g/beta shapes")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("Packed KDA requires batch size 1")
    if cu_seqlens is not None:
        offsets = cu_seqlens.tolist()
        if (cu_seqlens.ndim != 1 or len(offsets) < 2 or offsets[0] != 0
                or offsets[-1] != q.shape[1]
                or any(end <= start for start, end in zip(offsets, offsets[1:]))):
            raise ValueError("Packed KDA requires complete increasing boundaries")
    if q.shape[-1] > 256:
        raise ValueError("KDA requires K <= 256")
    if safe_gate and (lower_bound is None or not -5 <= lower_bound < 0):
        raise ValueError("Safe KDA requires -5 <= lower_bound < 0")
    assert initial_state is None
    assert use_qk_l2norm_in_kernel and use_gate_in_kernel
    heads, width = q.shape[-2:]
    assert A_log.shape == (heads,) and dt_bias.shape == (heads * width,)
    q_float, k_float = q.float(), k.float()
    q = (q_float * (q_float.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(q.dtype)
    k = (k_float * (k_float.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(k.dtype)
    output_dtype = q.dtype
    q, k, v, beta = q.float() * width ** -0.5, k.float(), v.float(), beta.float()
    raw_gate = g.float() + dt_bias.float().reshape(heads, width)
    rate = A_log.float().exp().reshape(heads, 1)
    decay = (-rate * F.softplus(raw_gate) if lower_bound is None
             else lower_bound * (rate * raw_gate).sigmoid())
    offsets = cu_seqlens.tolist() if cu_seqlens is not None else [0, q.shape[1]]
    outputs, states = [], []
    for start, end in zip(offsets, offsets[1:]):
        state = q.new_zeros(q.shape[0], heads, width, width)
        for token in range(start, end):
            state = state * decay[:, token].exp()[..., None]
            residual = v[:, token] - torch.einsum("bhk,bhkv->bhv", k[:, token], state)
            state = state + k[:, token, ..., None] * (beta[:, token, ..., None] * residual)[..., None, :]
            outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, token], state))
        states.append(state)
    output = torch.stack(outputs, 1).to(output_dtype)
    final_state = torch.cat(states, 0) if output_final_state else None
    return output, final_state


def cpu_sigmoid_norm(
    x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor,
    bias: torch.Tensor | None, activation: str = "swish",
    residual: torch.Tensor | None = None, prenorm: bool = False,
    residual_in_fp32: bool = False, eps: float = 1e-6,
) -> torch.Tensor:
    """FP32 sigmoid RMS normalization, storing once in the input dtype."""
    from torch.distributed.tensor import DTensor
    assert not any(isinstance(t, DTensor) for t in (x, g, weight, bias))
    assert bias is None and residual is None and not prenorm and not residual_in_fp32
    assert activation == "sigmoid"
    normalized = x.float() * (x.float().square().mean(-1, keepdim=True) + eps).rsqrt()
    return (normalized * weight.float() * g.float().sigmoid()).to(x.dtype)


def install_cpu_kda_bridge(monkeypatch: pytest.MonkeyPatch,
                           events: list[dict[str, Any]] | None = None) -> None:
    """Install unavailable KDA leaves before prepared closures capture imports."""
    import sys
    from types import SimpleNamespace
    from torchturbo.ops import get_or_create_ops
    from torchturbo.runtime.cpu import CpuRuntime
    dispatcher = get_or_create_ops(device_runtime=CpuRuntime())
    dispatch = dispatcher._dispatch
    def dispatch_with_conv(op_name, *args, **kwargs):
        if op_name == "causal_conv1d":
            if events is not None:
                events.append(dict(op=op_name, boundaries=kwargs["packed_params"].cu_seqlens_list,
                                   extent=args[0].shape[1]))
            return cpu_causal_conv1d(*args, **kwargs)
        return dispatch(op_name, *args, **kwargs)
    monkeypatch.setattr(dispatcher, "_dispatch", dispatch_with_conv)
    monkeypatch.setitem(sys.modules, "fla.modules.fused_norm_gate",
                        SimpleNamespace(rms_norm_gated=cpu_sigmoid_norm))
    monkeypatch.setitem(sys.modules, "fla.ops.kda", SimpleNamespace(chunk_kda=cpu_chunk_kda))


def cpu_vision_attention(module: torch.nn.Module, query: torch.Tensor,
                         key: torch.Tensor, value: torch.Tensor,
                         attention_mask: torch.Tensor | None, scaling: float,
                         dropout: float = 0.0, **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Noncausal FP32 segment SDPA for the actual VL vision FA2 leaf."""
    from torch.distributed.tensor import DTensor
    assert not any(isinstance(t, DTensor) for t in (query, key, value))
    assert query.shape == key.shape == value.shape
    assert query.dtype == key.dtype == value.dtype == torch.float32
    assert attention_mask is None and dropout == 0.0 and kwargs["is_causal"] is False
    bounds = kwargs["cu_seq_lens_q"].tolist()
    assert bounds == kwargs["cu_seq_lens_k"].tolist()
    assert bounds[0] == 0 and bounds[-1] == query.shape[2]
    assert all(end > start for start, end in zip(bounds, bounds[1:]))
    pieces = [F.scaled_dot_product_attention(query[:, :, start:end], key[:, :, start:end],
              value[:, :, start:end], is_causal=False, dropout_p=0.0, scale=scaling)
              for start, end in zip(bounds, bounds[1:])]
    return torch.cat(pieces, 2).transpose(1, 2).contiguous(), None


def install_cpu_vision_bridge(monkeypatch: pytest.MonkeyPatch,
                              events: list[dict[str, Any]] | None = None) -> None:
    """Dispatch only the concrete vision caller through its missing FA2 leaf."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from torchturbo.models.bailing_moe_v3_vl.modeling_bailing_moe_v3_vl import Qwen3VLMoeVisionAttention
    original = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    def attention(module, *args, **kwargs):
        if isinstance(module, Qwen3VLMoeVisionAttention):
            if events is not None:
                events.append(dict(op="vision_segment_attention", boundaries=kwargs["cu_seq_lens_q"].tolist()))
            return cpu_vision_attention(module, *args, **kwargs)
        return original(module, *args, **kwargs)
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._local_mapping, "flash_attention_2", attention)


def kda_oracle(module: torch.nn.Module, hidden: torch.Tensor,
               boundaries: torch.Tensor) -> torch.Tensor:
    """Literal full-weight per-token convolution and recurrent-state oracle."""
    offsets = boundaries.tolist()
    heads, width = module.num_heads, module.head_dim
    def linear(projection, value):
        return F.linear(value, projection.weight, projection.bias)
    def convolution(projection, conv):
        projected = linear(projection, hidden)
        tokens = []
        for start, end in zip(offsets, offsets[1:]):
            for token in range(start, end):
                convolved = torch.zeros_like(projected[:, token])
                for tap in range(conv.weight.shape[-1]):
                    source = token - conv.weight.shape[-1] + 1 + tap
                    if source >= start:
                        convolved = convolved + projected[:, source] * conv.weight[:, 0, tap]
                tokens.append(F.silu(convolved))
        return torch.stack(tokens, 1).reshape(*hidden.shape[:2], heads, width)
    query = convolution(module.q_proj, module.q_conv1d)
    key = convolution(module.k_proj, module.k_conv1d)
    value = convolution(module.v_proj, module.v_conv1d).float()
    query_float = query.float()
    key_float = key.float()
    query = (query_float / (query_float.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(query.dtype)
    key = (key_float / (key_float.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(key.dtype)
    beta = linear(module.b_proj, hidden).float().sigmoid()
    if module.no_kda_lora:
        raw_decay = linear(module.f_proj, hidden)
        output_gate = linear(module.g_proj, hidden)
    else:
        raw_decay = linear(module.f_b_proj, linear(module.f_a_proj, hidden))
        output_gate = linear(module.g_b_proj, linear(module.g_a_proj, hidden))
    raw_decay = raw_decay.float().reshape_as(query) + module.dt_bias.float().reshape(heads, width)
    decay_rate = module.A_log.float().exp()[:, None]
    if module.lower_bound is None:
        log_decay = -decay_rate * F.softplus(raw_decay)
    else:
        log_decay = module.lower_bound * torch.sigmoid(decay_rate * raw_decay)
    recurrent = []
    for start, end in zip(offsets, offsets[1:]):
        state = torch.zeros(hidden.shape[0], heads, width, width, dtype=torch.float32)
        for token in range(start, end):
            current_key = key[:, token].float()
            state = state * torch.exp(log_decay[:, token]).unsqueeze(-1)
            predicted = (current_key.unsqueeze(-1) * state).sum(-2)
            correction = beta[:, token].unsqueeze(-1) * (value[:, token] - predicted)
            state = state + current_key.unsqueeze(-1) * correction.unsqueeze(-2)
            recurrent.append((query[:, token].float().unsqueeze(-1) * state).sum(-2) * width ** -0.5)
    recurrent = torch.stack(recurrent, 1).to(query.dtype)
    normalized = recurrent.float() / (recurrent.float().square().mean(-1, keepdim=True) + module.o_norm.eps).sqrt()
    gated = normalized * module.o_norm.weight.float() * torch.sigmoid(output_gate.float().reshape_as(recurrent))
    return linear(module.o_proj, gated.to(recurrent.dtype).flatten(-2))
