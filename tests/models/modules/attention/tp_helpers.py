"""Narrow CPU leaf bridge for native tensor-parallel attention acceptance."""

import pytest
import torch


def install_cpu_attention_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve Ulysses and use the existing CPU max-logit attention provider."""
    from torchturbo.models.modules.attention.ulysses import ulysses
    from torchturbo.models.modules.attention.interface import attention_forward

    def attention(query, key, value, *args, **kwargs):
        assert not kwargs.pop("return_max_logits", False)
        assert all(not isinstance(tensor, torch.distributed.tensor.DTensor)
                   for tensor in (query, key, value))
        ratio = query.size(2) // key.size(2)
        key, value = key.repeat_interleave(ratio, 2), value.repeat_interleave(ratio, 2)
        output, _ = attention_forward(query, key, value, *args,
                                      return_max_logits=True, **kwargs)
        return output

    monkeypatch.setattr(ulysses, "attention_forward", attention)
