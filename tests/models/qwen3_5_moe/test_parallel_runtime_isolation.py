"""Public wrapper admission and isolation of two simultaneously live models."""

import gc
import weakref

import pytest
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from torchturbo.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from torchturbo.models.qwen3_5_moe.model import TurboQwen3_5MoeForConditionalGeneration
from torchturbo.engine.tensor_parallel.parallelize import register_tensor_parallel_cache_guard
from torchturbo.models.qwen3_5_moe_text.model import TurboQwen3_5MoeTextForCausalLM
from torchturbo.models.modules.attention.ulysses import ulysses

from .tp_helpers import text_config


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    for registry in (ALL_MASK_ATTENTION_FUNCTIONS, ALL_ATTENTION_FUNCTIONS):
        monkeypatch.setattr(type(registry), "_global_mapping", registry._global_mapping.copy())
        monkeypatch.setattr(registry, "_local_mapping", registry._local_mapping.copy())


@pytest.mark.parametrize("shared_config", [False, True])
@pytest.mark.parametrize("order", [(2, 1), (1, 2)])
def test_live_models_keep_their_full_attention_sp_binding(monkeypatch, shared_config, order):
    config = text_config()
    with torch.device("meta"):
        first = TurboQwen3_5MoeTextForCausalLM(config)
        second = TurboQwen3_5MoeTextForCausalLM(config if shared_config else text_config())
    groups = [object(), object()]
    calls = []

    def leaf(query, key, value, mask, length, **kwargs):
        calls.append((kwargs["sp_group"], kwargs["sp_size"]))
        return query

    monkeypatch.setattr(ulysses, "ulysses_flash_attention_forward", leaf)
    for model, group, size in zip((first, second), groups, order):
        model.on_parallelized({"sp": {"sp_size": size, "sp_group": group, "sp_rank": 0},
                               "tp": {"tp_size": 2}})
    for model in (first, second, first):
        attention = model.model.layers[1].self_attn
        query = torch.ones(1, 4, 2, 8, dtype=torch.bfloat16)
        entry = ALL_ATTENTION_FUNCTIONS[attention.config._attn_implementation]
        output, _ = entry(attention, query, query, query, None)
        assert output.shape == (1, 2, 4, 8)
    assert calls == [(groups[0], order[0]), (groups[1], order[1]),
                     (groups[0], order[0])]


@pytest.mark.parametrize("cache", [{"use_cache": True}, {"past_key_values": object()}])
def test_real_vlm_root_rejects_cache_before_embedding_or_vision(cache):
    config = Qwen3_5MoeConfig(
        text_config=text_config().to_dict(),
        vision_config={"depth": 1, "hidden_size": 8, "intermediate_size": 16,
                       "num_heads": 2, "out_hidden_size": 16},
    )
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = TurboQwen3_5MoeForConditionalGeneration(config)
    register_tensor_parallel_cache_guard(model)
    register_tensor_parallel_cache_guard(model.model.language_model)

    def computation(*args, **kwargs):
        raise AssertionError("Cache admission reached model computation")

    handles = [model.model.language_model.embed_tokens.register_forward_pre_hook(computation),
               model.model.visual.register_forward_pre_hook(computation)]
    try:
        with pytest.raises(ValueError, match="cache"):
            model(input_ids=torch.ones(1, 2, dtype=torch.long), **cache)
    finally:
        for handle in handles:
            handle.remove()


def test_runtime_bindings_do_not_serialize_or_retain_models():
    from torchturbo.models.modules.attention.ulysses import register_turbo_ulysses_attention

    model = TurboQwen3_5MoeTextForCausalLM(text_config())
    before = model.config.to_json_string()
    model.on_parallelized({"sp": {"sp_size": 1, "sp_group": None, "sp_rank": 0},
                           "tp": {"tp_size": 2}})
    assert model.config.to_json_string() == before
    reference = weakref.ref(model.model.layers[1].self_attn)
    del model
    gc.collect()
    assert reference() is None
    config = text_config()
    config_id = id(config)
    direct_before = config.to_json_string()
    register_turbo_ulysses_attention(config, object(), 2)
    assert config.to_json_string() == direct_before
    assert config_id in ulysses._config_attention_bindings
    del config
    gc.collect()
    assert config_id not in ulysses._config_attention_bindings
