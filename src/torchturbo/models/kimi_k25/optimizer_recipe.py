"""Kimi K2.5 parameter semantics for TorchTurbo's distributed Muon."""

from __future__ import annotations

from torch import nn
from torch.distributed.tensor import Shard

from torchturbo.engine.optimizers import (
    ExcludeFromMuon,
    MLAQKClipSpec,
    MuonParameterOverride,
    MuonParameterSpec,
    MuonRecipe,
    RowSplitComputeView,
)


def get_muon_recipe(model: nn.Module) -> MuonRecipe:
    """Describe the Kimi K2.5 parameters that differ from the Muon defaults.

    Only the text transformer trains with Muon: the vision tower and the
    projector fall back to AdamW. MLA splits Q-NoPE/Q-RoPE and K-NoPE/V within
    each head, and separates KV latent rows from shared K-RoPE rows. The backend
    assigns each complete matrix to a compute rank.
    Fused expert layouts need no entry here because ``GroupedMLP``
    declares them itself, and neither do PEFT adapters because the engine never
    routes them to Muon.

    Patterns follow the configured module structure so unmatched-pattern
    warnings remain actionable.
    """
    text_config = model.config.text_config
    mla_query = MuonParameterSpec(
        compute_placement=Shard(0),
        compute_view=RowSplitComputeView(
            text_config.num_attention_heads,
            (text_config.qk_nope_head_dim, text_config.qk_rope_head_dim),
        ),
    )
    mla_kv = MuonParameterSpec(
        compute_placement=Shard(0),
        compute_view=RowSplitComputeView(
            text_config.num_attention_heads,
            (text_config.qk_nope_head_dim, text_config.v_head_dim),
        ),
    )
    mla_kv_down = MuonParameterSpec(
        compute_placement=Shard(0),
        compute_view=RowSplitComputeView(
            1, (text_config.kv_lora_rank, text_config.qk_rope_head_dim)
        ),
    )
    # Attention builds either q_proj or the q_a_proj/q_b_proj pair.
    query_projection = "q_proj" if text_config.q_lora_rank is None else "q_b_proj"
    parameter_overrides: dict[str, MuonParameterOverride] = {
        f"language_model.model.layers.*.self_attn.{query_projection}.weight": mla_query,
        "language_model.model.layers.*.self_attn.kv_b_proj.weight": mla_kv,
        "language_model.model.layers.*.self_attn.kv_a_proj_with_mqa.weight": mla_kv_down,
        "vision_tower.**": ExcludeFromMuon(),
    }
    # An identity projector holds no parameters, so it has nothing to exclude.
    if model.config.vision_config.mm_projector_type != "identity":
        parameter_overrides["mm_projector.**"] = ExcludeFromMuon()
    qk_clip_specs: dict[str, MLAQKClipSpec] = {
        "language_model.model.layers.*.self_attn": MLAQKClipSpec(
            query_projection=query_projection,
            key_value_projection="kv_b_proj",
            num_heads=int(text_config.num_attention_heads),
            qk_nope_head_dim=int(text_config.qk_nope_head_dim),
            qk_rope_head_dim=int(text_config.qk_rope_head_dim),
            v_head_dim=int(text_config.v_head_dim),
        )
    }
    return MuonRecipe(
        parameter_overrides=parameter_overrides,
        qk_clip_specs=qk_clip_specs,
    )
