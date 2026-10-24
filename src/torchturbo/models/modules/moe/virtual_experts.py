"""MoonEP-style bounded virtual expert placement and route remapping.

The placement algorithm follows Megatron-LM's virtual_expert_triton.py:
https://github.com/NVIDIA/Megatron-LM
"""

import torch


def plan_expert_placement(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Balance CPU [source_rank, logical_expert] counts without dropping routes.

    Returns:
        Allocations [logical_expert, destination_rank] and logical expert IDs
        [destination_rank, replica_slot], with -1 marking unused replica slots.
        Each rank keeps its native experts and has equally many replica slots.
    """
    num_ranks, num_experts = counts.shape
    num_local_experts = num_experts // num_ranks
    totals = counts.sum(dim=0)
    num_routes = int(totals.sum())
    targets = torch.full((num_ranks,), num_routes // num_ranks, dtype=torch.long)
    targets[:num_routes % num_ranks] += 1
    balances = (totals.reshape(num_ranks, num_local_experts).sum(dim=1) - targets).tolist()
    quotas = [[0] * num_ranks for _ in range(num_ranks)]
    while max(balances) > 0:
        sender = max(range(num_ranks), key=balances.__getitem__)
        receiver = min(range(num_ranks), key=balances.__getitem__)
        # Taking the whole deficit gives each receiver exactly one sender's
        # experts, so its replicas fit within num_local_experts slots.
        moved = -balances[receiver]
        quotas[sender][receiver] = moved
        balances[sender] -= moved
        balances[receiver] = 0

    allocations = torch.zeros((num_experts, num_ranks), dtype=torch.long)
    experts = torch.arange(num_experts)
    allocations[experts, experts // num_local_experts] = totals
    for owner, quota in enumerate(quotas):
        start = owner * num_local_experts
        remaining = totals[start:start + num_local_experts].tolist()
        while max(quota) > 0:
            destination = max(range(num_ranks), key=quota.__getitem__)
            local_expert = max(range(num_local_experts), key=remaining.__getitem__)
            moved = min(quota[destination], remaining[local_expert])
            allocations[start + local_expert, owner] -= moved
            allocations[start + local_expert, destination] += moved
            remaining[local_expert] -= moved
            quota[destination] -= moved

    copies = torch.full((num_ranks, num_local_experts), -1, dtype=torch.long)
    for destination in range(num_ranks):
        remote = [
            (count, expert) for expert, count in enumerate(allocations[:, destination].tolist())
            if count > 0 and expert // num_local_experts != destination
        ]
        # Match the reference's highest-expert-ID tie break for replica slots.
        remote.sort(reverse=True)
        assert len(remote) <= num_local_experts, "Virtual expert placement exceeded replica slots"
        copies[destination, :len(remote)] = torch.tensor(
            [expert for _, expert in remote], dtype=torch.long,
        )
    return allocations, copies


def remap_expert_indices(
    indices: torch.Tensor, counts: torch.Tensor, allocations: torch.Tensor,
    copies: torch.Tensor, rank: int,
) -> torch.Tensor:
    """Map [tokens, topk] logical IDs while preserving their probability order.

    Occurrences of an expert are ordered by source rank, then local flattened
    route order. Counts must be the histogram of these IDs from the same forward
    on each rank. Counts and placement tables are on CPU; IDs stay on the input
    device. The caller sorts returned IDs together with selected probabilities
    before constructing a runtime RoutingResult.
    """
    num_ranks, num_local_experts = copies.shape
    num_experts = allocations.shape[0]
    flat_indices = indices.flatten()
    order = flat_indices.argsort(stable=True)
    sorted_experts = flat_indices[order]
    local_counts = counts[rank].to(indices.device)
    local_starts = local_counts.cumsum(0) - local_counts
    source_prefix = counts[:rank].sum(dim=0).to(indices.device)
    ordinals = (
        torch.arange(flat_indices.numel(), device=indices.device)
        - local_starts[sorted_experts] + source_prefix[sorted_experts]
    )

    # Concatenate experts' ordinal ranges so one search avoids a routes-by-ranks
    # temporary. Right insertion skips empty destination ranges at each boundary.
    totals = counts.sum(dim=0)
    expert_starts = totals.cumsum(0) - totals
    boundaries = (allocations.cumsum(1) + expert_starts[:, None]).flatten().to(indices.device)
    global_ordinals = ordinals + expert_starts.to(indices.device)[sorted_experts]
    destinations = torch.searchsorted(boundaries, global_ordinals, right=True) % num_ranks

    slots = torch.full((num_experts, num_ranks), -1, dtype=torch.long)
    experts = torch.arange(num_experts)
    slots[experts, experts // num_local_experts] = experts % num_local_experts
    destinations_with_copies, replica_slots = torch.where(copies >= 0)
    slots[copies[destinations_with_copies, replica_slots], destinations_with_copies] = (
        num_local_experts + replica_slots
    )
    slots = slots.to(indices.device)
    sorted_runtime = destinations * (2 * num_local_experts) + slots[sorted_experts, destinations]
    runtime_indices = torch.empty_like(flat_indices)
    runtime_indices[order] = sorted_runtime
    return runtime_indices.reshape_as(indices)
