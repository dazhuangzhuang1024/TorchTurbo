"""Placement conservation and virtual expert route identity contracts."""

import pytest
import torch

from torchturbo.models.modules.moe.virtual_experts import (
    plan_expert_placement,
    remap_expert_indices,
)


def test_full_deficit_transfer_can_exceed_sender_excess():
    counts = torch.tensor([
        [6, 4, 5, 4, 2, 2, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    allocations, copies = plan_expert_placement(counts)
    torch.testing.assert_close(allocations, torch.tensor([
        [1, 0, 0, 5], [4, 0, 0, 0], [0, 3, 2, 0], [1, 3, 0, 0],
        [0, 0, 2, 0], [0, 0, 2, 0], [0, 0, 0, 1], [0, 0, 0, 0],
    ]))
    torch.testing.assert_close(copies, torch.tensor([
        [3, -1], [-1, -1], [2, -1], [0, -1],
    ]))


def test_rank_and_expert_ties_choose_lowest_id():
    counts = torch.tensor([[2, 2, 2, 2, 0, 0], [0] * 6, [0] * 6])
    allocations, copies = plan_expert_placement(counts)
    torch.testing.assert_close(allocations, torch.tensor([
        [0, 0, 2], [2, 0, 0], [1, 1, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0],
    ]))
    torch.testing.assert_close(copies, torch.tensor([[2, -1], [-1, -1], [0, -1]]))


def test_replica_slot_ties_choose_highest_global_expert():
    counts = torch.tensor([[2, 2, 2, 2, 0, 0, 0, 0], [0] * 8])
    allocations, copies = plan_expert_placement(counts)
    torch.testing.assert_close(allocations, torch.tensor([
        [0, 2], [0, 2], [2, 0], [2, 0], [0, 0], [0, 0], [0, 0], [0, 0],
    ]))
    torch.testing.assert_close(copies, torch.tensor([[-1, -1, -1, -1], [1, 0, -1, -1]]))


def test_floor_ceil_targets_and_empty_experts():
    counts = torch.tensor([[5, 0, 0, 0], [0, 0, 0, 0]])
    allocations, copies = plan_expert_placement(counts)
    torch.testing.assert_close(allocations, torch.tensor([[3, 2], [0, 0], [0, 0], [0, 0]]))
    torch.testing.assert_close(copies, torch.tensor([[-1, -1], [0, -1]]))


@pytest.mark.parametrize("num_ranks,num_local_experts", [(1, 3), (2, 2), (4, 1)])
def test_all_empty_placement(num_ranks, num_local_experts):
    num_experts = num_ranks * num_local_experts
    counts = torch.zeros((num_ranks, num_experts), dtype=torch.long)
    allocations, copies = plan_expert_placement(counts)
    torch.testing.assert_close(allocations, torch.zeros((num_experts, num_ranks), dtype=torch.long))
    torch.testing.assert_close(copies, torch.full((num_ranks, num_local_experts), -1))
    for rank in range(num_ranks):
        indices = torch.empty((0, 2), dtype=torch.long)
        mapped = remap_expert_indices(indices, counts, allocations, copies, rank)
        assert mapped.shape == indices.shape


def test_remap_preserves_source_rank_and_flattened_topk_order():
    counts = torch.tensor([[2, 0, 1, 1], [1, 0, 0, 1]])
    allocations, copies = plan_expert_placement(counts)
    torch.testing.assert_close(allocations, torch.tensor([[3, 0], [0, 0], [0, 1], [0, 2]]))
    torch.testing.assert_close(copies, torch.tensor([[-1, -1], [-1, -1]]))
    first = torch.tensor([[0, 3], [2, 0]])
    second = torch.tensor([[3, 0]])
    torch.testing.assert_close(
        remap_expert_indices(first, counts, allocations, copies, 0),
        torch.tensor([[0, 5], [4, 0]]),
    )
    torch.testing.assert_close(
        remap_expert_indices(second, counts, allocations, copies, 1),
        torch.tensor([[5, 0]]),
    )


def test_remap_splits_expert_occurrences_across_sources():
    counts = torch.tensor([[3, 0, 0, 1], [2, 0, 0, 0]])
    allocations, copies = plan_expert_placement(counts)
    torch.testing.assert_close(allocations, torch.tensor([[3, 2], [0, 0], [0, 0], [0, 1]]))
    torch.testing.assert_close(copies, torch.tensor([[-1, -1], [0, -1]]))
    first = torch.tensor([[0, 3], [0, 0]])
    second = torch.tensor([[0, 0]])
    torch.testing.assert_close(
        remap_expert_indices(first, counts, allocations, copies, 0),
        torch.tensor([[0, 5], [0, 0]]),
    )
    torch.testing.assert_close(
        remap_expert_indices(second, counts, allocations, copies, 1),
        torch.tensor([[6, 6]]),
    )


@pytest.mark.parametrize("num_ranks,num_local_experts", [(1, 3), (2, 1), (3, 2), (4, 3), (8, 2)])
def test_random_placement_conserves_routes_and_bounds_replica_slots(num_ranks, num_local_experts):
    generator = torch.Generator().manual_seed(781)
    num_experts = num_ranks * num_local_experts
    for _ in range(50):
        counts = torch.randint(0, 13, (num_ranks, num_experts), generator=generator)
        counts[:, torch.randperm(num_experts, generator=generator)[:num_experts // 2]] = 0
        allocations, copies = plan_expert_placement(counts)
        torch.testing.assert_close(allocations.sum(dim=1), counts.sum(dim=0))
        assert (allocations >= 0).all()
        total = int(counts.sum())
        targets = torch.full((num_ranks,), total // num_ranks)
        targets[:total % num_ranks] += 1
        torch.testing.assert_close(allocations.sum(dim=0), targets)
        for rank in range(num_ranks):
            remote = torch.nonzero(allocations[:, rank] > 0).flatten()
            remote = remote[remote // num_local_experts != rank]
            selected = copies[rank][copies[rank] >= 0]
            assert len(selected) <= num_local_experts
            torch.testing.assert_close(selected.sort().values, remote.sort().values)
            assert len(selected.unique()) == len(selected)


@pytest.mark.parametrize("topk", [1, 2, 4])
def test_random_remap_keeps_logical_identity_and_exact_assignment_counts(topk):
    generator = torch.Generator().manual_seed(97 + topk)
    num_ranks, num_local_experts = 4, 2
    num_experts = num_ranks * num_local_experts
    for _ in range(20):
        source_indices = [
            torch.rand((num_tokens, num_experts), generator=generator).argsort(dim=1)[:, :topk]
            for num_tokens in (0, 1, 7, 3)
        ]
        counts = torch.stack([
            torch.bincount(indices.flatten(), minlength=num_experts) for indices in source_indices
        ])
        allocations, copies = plan_expert_placement(counts)
        assigned = torch.zeros_like(allocations)
        runtime_to_logical = torch.cat([
            torch.cat((torch.arange(rank * num_local_experts, (rank + 1) * num_local_experts), copies[rank]))
            for rank in range(num_ranks)
        ])
        for rank, indices in enumerate(source_indices):
            mapped = remap_expert_indices(indices, counts, allocations, copies, rank)
            assert mapped.shape == indices.shape
            assert mapped.device == indices.device
            assert (mapped >= 0).all()
            assert (mapped < num_ranks * 2 * num_local_experts).all()
            assert (mapped.sort(dim=1).values.diff(dim=1) > 0).all()
            torch.testing.assert_close(runtime_to_logical[mapped], indices)
            destinations = mapped // (2 * num_local_experts)
            cells = indices.flatten() * num_ranks + destinations.flatten()
            assigned += torch.bincount(cells, minlength=num_experts * num_ranks).reshape_as(assigned)
        torch.testing.assert_close(assigned, allocations)
