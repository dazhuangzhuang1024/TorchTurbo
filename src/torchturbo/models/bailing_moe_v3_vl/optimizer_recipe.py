"""Bailing V3 VL parameter semantics for TorchTurbo's distributed Muon."""

from __future__ import annotations

from torch import nn
from torch.distributed.tensor import Shard

from torchturbo.engine.optimizers import (
    AttentionPerHeadComputeView,
    ExcludeFromMuon,
    MLAQKClipSpec,
    MuonParameterOverride,
    MuonParameterSpec,
    MuonRecipe,
    RowSplitComputeView,
)

from .modeling_bailing_moe_v3_vl import BailingMoeV3MultiLatentAttention


def get_muon_recipe(model: nn.Module) -> MuonRecipe:
    """Describe text attention layouts while keeping vision parameters on AdamW.

    MLA splits Q-NoPE/Q-RoPE and K-NoPE/V within each head, and separates
    KV latent rows from shared K-RoPE rows. KDA and elementwise gates keep one
    matrix per head; scalar gates use AdamW. Query input projections keep the
    whole matrix layout, and fused experts declare their layouts in GroupedMLP.
    MTP layers share the text decoder's namespace and use MLA semantics.
    """
    text_config = model.config.text_config
    per_head = MuonParameterSpec(
        compute_placement=Shard(0),
        compute_view=AttentionPerHeadComputeView(text_config.num_attention_heads),
    )
    query_projection = "q_proj" if text_config.q_lora_rank is None else "q_b_proj"
    mla_specs = {
        query_projection: MuonParameterSpec(
            compute_placement=Shard(0),
            compute_view=RowSplitComputeView(
                text_config.num_attention_heads,
                (text_config.qk_nope_head_dim, text_config.qk_rope_head_dim),
            ),
        ),
        "kv_b_proj": MuonParameterSpec(
            compute_placement=Shard(0),
            compute_view=RowSplitComputeView(
                text_config.num_attention_heads,
                (text_config.qk_nope_head_dim, text_config.v_head_dim),
            ),
        ),
        "kv_a_proj_with_mqa": MuonParameterSpec(
            compute_placement=Shard(0),
            compute_view=RowSplitComputeView(
                1, (text_config.kv_lora_rank, text_config.qk_rope_head_dim)
            ),
        ),
    }
    kda_gate_suffix = "proj" if text_config.no_kda_lora else "b_proj"
    parameter_overrides: dict[str, MuonParameterOverride] = {
        "model.visual.**": ExcludeFromMuon(),
        "linear_proj.**": ExcludeFromMuon(),
    }
    qk_clip_specs: dict[str, MLAQKClipSpec] = {}
    for layer_index, layer in enumerate(model.model.layers):
        attention_fqn = f"model.layers.{layer_index}.attention"
        if isinstance(layer.attention, BailingMoeV3MultiLatentAttention):
            for projection, spec in mla_specs.items():
                parameter_overrides[f"{attention_fqn}.{projection}.weight"] = spec
            projections = []
            gate_type = text_config.gated_attention_proj_granularity_type
            if gate_type == "head_wise":
                parameter_overrides[f"{attention_fqn}.g_proj.weight"] = ExcludeFromMuon()
            elif gate_type == "element_wise":
                projections.append("g_proj")
            # KDA has no per-head QK logits to observe, so only MLA is clipped.
            qk_clip_specs[attention_fqn] = MLAQKClipSpec(
                query_projection=query_projection,
                key_value_projection="kv_b_proj",
                num_heads=text_config.num_attention_heads,
                qk_nope_head_dim=text_config.qk_nope_head_dim,
                qk_rope_head_dim=text_config.qk_rope_head_dim,
                v_head_dim=text_config.v_head_dim,
            )
        else:
            projections = [
                "q_proj",
                "k_proj",
                "v_proj",
                f"f_{kda_gate_suffix}",
                f"g_{kda_gate_suffix}",
            ]
            parameter_overrides[f"{attention_fqn}.b_proj.weight"] = ExcludeFromMuon()
        for projection in projections:
            parameter_overrides[f"{attention_fqn}.{projection}.weight"] = per_head
    return MuonRecipe(
        parameter_overrides=parameter_overrides,
        qk_clip_specs=qk_clip_specs,
    )
