"""Tests for shared causal targets and the FSDP2 target hook."""

from __future__ import annotations

import pytest
import torch

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.models.modules import loss


@pytest.mark.parametrize(
    "labels,cu_seq_lens,shift,ignore_index,expected",
    [
        pytest.param(
            [[1, 2, 3], [4, 5, 6]],
            None,
            1,
            -100,
            [[2, 3, -100], [5, 6, -100]],
            id="ordinary-rows",
        ),
        pytest.param(
            [[1, -100, 3, -100]],
            None,
            1,
            -100,
            [[-100, 3, -100, -100]],
            id="ignored-labels",
        ),
        pytest.param(
            [[1, 2, 3, 4, 5, 6]],
            [0, 2, 5, 6],
            1,
            -100,
            [[2, -100, 4, 5, -100, -100]],
            id="packed-boundaries",
        ),
        pytest.param(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            None,
            2,
            -100,
            [[3, 4, 5, -100, -100], [8, 9, 10, -100, -100]],
            id="mtp-shift",
        ),
        pytest.param(
            [[1, 2, 3, 4, 5, 6]],
            [0, 1, 4, 6],
            2,
            -100,
            [[-100, 4, -100, -100, -100, -100]],
            id="packed-mtp-shift",
        ),
        pytest.param(
            [[1, 2, -7, 4]],
            None,
            1,
            -7,
            [[2, -7, 4, -7]],
            id="custom-ignore-index",
        ),
        pytest.param(
            [[1, 2, 3, 4, 5, 6, 70, 80]],
            [0, 3, 6],
            1,
            -100,
            [[2, 3, -100, 5, 6, -100, -100, -100]],
            id="packed-padded-tail",
        ),
        pytest.param(
            [[-100, -100, -100]],
            [0, 3],
            1,
            -100,
            [[-100, -100, -100]],
            id="fully-ignored",
        ),
    ],
)
def test_prepare_loss_targets_matches_expected_rows(
    labels, cu_seq_lens, shift, ignore_index, expected
):
    labels_tensor = torch.tensor(labels)
    boundaries = (
        torch.tensor(cu_seq_lens, dtype=torch.int32)
        if cu_seq_lens is not None
        else None
    )
    shared_targets = loss.prepare_loss_targets(
        labels_tensor, boundaries, shift, ignore_index
    )
    backend_targets = FSDP2Backend.prepare_loss_targets(
        labels_tensor, boundaries, shift, ignore_index
    )

    torch.testing.assert_close(shared_targets, torch.tensor(expected))
    torch.testing.assert_close(backend_targets, shared_targets)
    torch.testing.assert_close(labels_tensor, torch.tensor(labels))


def test_prepare_loss_targets_defaults_preserve_noncontiguous_input():
    labels = torch.tensor([[1, 2], [3, 4], [5, 6]]).transpose(0, 1)
    assert not labels.is_contiguous()

    targets = loss.prepare_loss_targets(labels)
    backend_targets = FSDP2Backend.prepare_loss_targets(labels)

    torch.testing.assert_close(targets, torch.tensor([[3, 5, -100], [4, 6, -100]]))
    torch.testing.assert_close(backend_targets, targets)
    torch.testing.assert_close(labels, torch.tensor([[1, 3, 5], [2, 4, 6]]))
    assert targets.is_contiguous()
