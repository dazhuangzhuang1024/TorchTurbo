"""Language-model target preparation shared by training integrations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["prepare_loss_targets"]


def prepare_loss_targets(
    labels: torch.Tensor,
    cu_seq_lens: torch.Tensor | None = None,
    shift: int = 1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build full-row targets with causal shift and packed boundary masking."""
    targets = F.pad(labels, (0, shift), value=ignore_index)
    targets = targets[..., shift:].contiguous()
    if cu_seq_lens is not None:
        positions = torch.arange(labels.size(-1), device=labels.device)
        segment_index = torch.bucketize(
            positions, cu_seq_lens[1:], right=True
        ).clamp(max=cu_seq_lens.numel() - 2)
        sequence_ends = cu_seq_lens[segment_index + 1]
        targets.masked_fill_(positions + shift >= sequence_ends, ignore_index)
    return targets
