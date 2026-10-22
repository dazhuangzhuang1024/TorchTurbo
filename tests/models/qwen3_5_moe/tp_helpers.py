"""CPU mathematical leaves and an independent Qwen3.5 GDN oracle."""

import torch
from torch.nn import functional as F

from torchturbo.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule
from torchturbo.models.qwen3_5_moe_text.model import TurboQwen3_5MoeTextForCausalLM


def text_config(**options):
    values = dict(
        vocab_size=24, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=4, linear_num_value_heads=8,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=3, moe_intermediate_size=12,
        shared_expert_intermediate_size=12, num_experts=4,
        num_experts_per_tok=2, use_cache=False, dtype=torch.float32,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": .5, "mrope_section": [1, 1, 0]},
    )
    values.update(options)
    config = Qwen3_5MoeTextConfig(**values)
    config._attn_implementation = "eager"
    return config


def text_model(**options):
    return TurboQwen3_5MoeTextForCausalLM(text_config(**options))


def initialize_gdn(module, *, cancellation_sensitive=False):
    """Distinct channel weights keep every head and gate gradient observable."""
    with torch.no_grad():
        generator = torch.Generator().manual_seed(23)
        for index, (name, parameter) in enumerate(module.named_parameters()):
            numbers = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            if name == "A_log":
                parameter.copy_(torch.linspace(-1.2, -.4, parameter.numel()))
            elif name == "dt_bias":
                parameter.copy_(torch.linspace(-.3, .3, parameter.numel()))
            elif name == "norm.weight":
                parameter.copy_(torch.linspace(.7, 1.3, parameter.numel()))
            elif cancellation_sensitive:
                parameter.copy_(.15 * torch.sin(numbers * .17 + index * .31))
            elif name == "conv1d.weight":
                parameter.copy_(.12 * torch.sin(numbers * .17 + index * .31))
                parameter[:, 0, -1].add_(.5)
            else:
                parameter.copy_(.12 * torch.randn(parameter.shape, generator=generator))


def install_cpu_leaves(monkeypatch, *, events=None):
    """Retain public dispatcher normalization and replace unavailable leaves."""
    import torchturbo.ops as turbo_ops
    from torchturbo.ops.dispatcher import OpDispatcher

    dispatcher = OpDispatcher.__new__(OpDispatcher)

    def dispatch(op_name, *args, **kwargs):
        if op_name not in ("causal_conv1d", "gated_delta_rule"):
            return turbo_ops.get_ops()._dispatch(op_name, *args, **kwargs)
        assert all(not isinstance(value, torch.distributed.tensor.DTensor)
                   for value in args if isinstance(value, torch.Tensor))
        boundaries = kwargs["packed_params"].cu_seqlens_list
        if events is not None:
            events.append(dict(op=op_name, metadata=kwargs["packed_params"],
                               boundaries=list(boundaries), dtypes=[str(value.dtype) for value in args if isinstance(value, torch.Tensor)]))
        if op_name == "causal_conv1d":
            x, weight, bias = args[:3]
            outputs = []
            for start, end in zip(boundaries, boundaries[1:]):
                sequence = x[:, start:end].transpose(1, 2)
                output = F.conv1d(sequence, weight.unsqueeze(1), bias,
                                  padding=weight.size(-1) - 1, groups=x.size(-1))
                outputs.append(F.silu(output[..., :end-start]).transpose(1, 2))
            return torch.cat(outputs, dim=1), None
        q, k, v, g, beta = args[:5]
        outputs = []
        for start, end in zip(boundaries, boundaries[1:]):
            output, _ = torch_chunk_gated_delta_rule(
                q[:, start:end], k[:, start:end], v[:, start:end],
                g[:, start:end], beta[:, start:end], chunk_size=4,
                use_qk_l2norm_in_kernel=True,
            )
            outputs.append(output)
        return torch.cat(outputs, dim=1), None

    monkeypatch.setattr(dispatcher, "_dispatch", dispatch)
    monkeypatch.setattr(turbo_ops, "ops", dispatcher)


def gdn_oracle(module, x, boundaries, *, qkv_projection=None):
    """Literal segment convolution and token recurrence, without model helpers."""
    projected = (F.linear(x, module.in_proj_qkv.weight)
                 if qkv_projection is None else qkv_projection(x, module.in_proj_qkv.weight))
    channels = []
    for channel in range(projected.size(-1)):
        segments = []
        for start, end in zip(boundaries, boundaries[1:]):
            tokens = []
            for token in range(start, end):
                terms = []
                for offset in range(module.conv_kernel_size):
                    source = token - module.conv_kernel_size + 1 + offset
                    if source >= start:
                        terms.append(projected[:, source, channel].float() * module.conv1d.weight[channel, 0, offset].float())
                tokens.append(F.silu(torch.stack(terms).sum(0).to(projected.dtype)))
            segments.append(torch.stack(tokens, dim=1))
        channels.append(torch.cat(segments, dim=1))
    convolved = torch.stack(channels, dim=-1)
    q, k, v = convolved.split((module.key_dim, module.key_dim, module.value_dim), -1)
    q = q.reshape(*x.shape[:2], module.num_k_heads, module.head_k_dim)
    k = k.reshape_as(q)
    v = v.reshape(*x.shape[:2], module.num_v_heads, module.head_v_dim)
    ratio = module.num_v_heads // module.num_k_heads
    q = q.repeat_interleave(ratio, 2)
    k = k.repeat_interleave(ratio, 2)
    q = (q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)).float()
    k = (k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)).float()
    q = q * module.head_k_dim ** -.5
    a = F.linear(x, module.in_proj_a.weight)
    beta = F.linear(x, module.in_proj_b.weight).sigmoid().float()
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    outputs = []
    for start, end in zip(boundaries, boundaries[1:]):
        state = x.new_zeros(x.size(0), module.num_v_heads,
                            module.head_k_dim, module.head_v_dim).float()
        for token in range(start, end):
            state = state * g[:, token].exp()[..., None, None]
            prediction = torch.einsum("bhd,bhdv->bhv", k[:, token], state)
            delta = (v[:, token].float() - prediction) * beta[:, token, :, None]
            state = state + k[:, token, :, :, None] * delta[:, :, None, :]
            outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, token], state))
    core = torch.stack(outputs, dim=1).to(x.dtype)
    z = F.linear(x, module.in_proj_z.weight).reshape_as(core)
    normalized = core.float() * torch.rsqrt(core.float().square().mean(-1, keepdim=True) + module.layer_norm_epsilon)
    normalized = module.norm.weight * normalized.to(x.dtype)
    normalized = (normalized * F.silu(z.float())).to(x.dtype)
    return F.linear(normalized.flatten(2), module.out_proj.weight)


def model_style(plan):
    return plan.tensor_parallel_plan.plan[""]


def model_validator(plan, *, tp_size=2, sp_size=1):
    plan.tp_size = plan.etp_size = tp_size
    plan.sp_size = sp_size
    return plan.tensor_parallel_plan.validator, plan
