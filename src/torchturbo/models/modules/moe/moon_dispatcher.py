"""MoonEP virtual expert balancing over the ordinary all-to-all transport."""

from dataclasses import replace

import torch
import torch.distributed as dist

from ..comm.comm import all_gather_tensor
from .moe_config import MoEConfig
from .token_dispatcher import MoEAlltoAllTokenDispatcher
from .utils import RoutingResult
from .virtual_expert_weights import VirtualExpertWeightPlan
from .virtual_experts import plan_expert_placement, remap_expert_indices


class MoEMoonTokenDispatcher(MoEAlltoAllTokenDispatcher):
    """Balance logical expert routes across native and temporary replica slots."""

    def __init__(
        self, num_local_experts: int,
        config: MoEConfig, ep_group: dist.ProcessGroup,
    ) -> None:
        self.num_logical_experts = config.num_moe_experts
        self.ep_rank = dist.get_rank(ep_group)
        num_runtime_experts = 2 * num_local_experts
        runtime_config = replace(
            config, num_moe_experts=2 * config.num_moe_experts,
            expert_model_parallel_size=dist.get_world_size(ep_group),
        )
        start = self.ep_rank * num_runtime_experts
        super().__init__(
            num_runtime_experts, list(range(start, start + num_runtime_experts)),
            runtime_config, ep_group,
        )
        self.expert_plan: VirtualExpertWeightPlan | None = None

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing: RoutingResult,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan logical routes and prepare weights without touching parameters."""
        local_counts = torch.bincount(
            routing.expert_indices.flatten(), minlength=self.num_logical_experts,
        )
        counts = all_gather_tensor(local_counts, self.ep_group).reshape(
            self.ep_size, self.num_logical_experts,
        ).cpu()
        allocations, copies = plan_expert_placement(counts)
        runtime_indices = remap_expert_indices(
            routing.expert_indices, counts, allocations, copies, self.ep_rank,
        )
        runtime_indices, order = runtime_indices.sort(dim=1)
        selected_probs = routing.selected_probs().gather(1, order)
        runtime_probs = selected_probs.new_zeros(
            (runtime_indices.shape[0], self.num_experts)
        ).scatter(1, runtime_indices, selected_probs)
        runtime_map = torch.zeros_like(runtime_probs, dtype=torch.bool).scatter_(
            1, runtime_indices, True,
        )
        self.expert_plan = VirtualExpertWeightPlan.from_copies(
            copies, self.ep_group, hidden_states.device,
        )
        runtime_routing = RoutingResult(runtime_probs, runtime_map, runtime_indices)
        return super().dispatch_preprocess(hidden_states, runtime_routing)
