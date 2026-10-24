"""Per-forward virtual weights with ordered EP gradient reduction."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..comm.comm import all_to_all


class _ExchangeExpertWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packet, hidden_states, probs, group, receive_splits, send_splits):
        ctx.group = group
        ctx.receive_splits = receive_splits
        ctx.send_splits = send_splits
        received = all_to_all(group, packet, receive_splits, send_splits)
        return received, hidden_states, probs

    @staticmethod
    def backward(ctx, grad_received, grad_hidden, grad_probs):
        # The input/probability edges hold token-dispatch backward until weight
        # reduction finishes, including on ranks with no local replica work.
        grad_packet = all_to_all(
            ctx.group, grad_received, ctx.send_splits, ctx.receive_splits,
        )
        return grad_packet, grad_hidden, grad_probs, None, None, None


@dataclass(frozen=True)
class VirtualExpertWeightPlan:
    """Selected weight transfers for one forward and its checkpoint replays.

    Tensor indices are immutable metadata; all parameter access happens inside
    expert forward. Every EP rank must participate in the same backward calls.
    """

    group: dist.ProcessGroup
    send_indices: torch.Tensor
    send_splits: list[int]
    receive_slots: torch.Tensor
    receive_splits: list[int]

    @classmethod
    def from_copies(
        cls, copies: torch.Tensor, group: dist.ProcessGroup, device: torch.device,
    ) -> "VirtualExpertWeightPlan":
        """Order selected logical IDs by peer and destination replica slot."""
        ep_size, num_local_experts = copies.shape
        rank = dist.get_rank(group)
        copy_ids = copies.tolist()
        send_indices = []
        send_splits = []
        for destination_copies in copy_ids:
            selected = [
                expert % num_local_experts for expert in destination_copies
                if expert >= 0 and expert // num_local_experts == rank
            ]
            send_indices.extend(selected)
            send_splits.append(len(selected))
        receive_slots = []
        receive_splits = []
        for owner in range(ep_size):
            slots = [
                slot for slot, expert in enumerate(copy_ids[rank])
                if expert >= 0 and expert // num_local_experts == owner
            ]
            receive_slots.extend(slots)
            receive_splits.append(len(slots))
        return cls(
            group=group,
            send_indices=torch.tensor(send_indices, dtype=torch.long, device=device),
            send_splits=send_splits,
            receive_slots=torch.tensor(receive_slots, dtype=torch.long, device=device),
            receive_splits=receive_splits,
        )

    def materialize(
        self, weight1: torch.Tensor, weight2: torch.Tensor,
        hidden_states: torch.Tensor, probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return native-plus-replica matrices and ordered expert input edges.

        Args:
            weight1: Unsharded native FC1 matrices, [local_experts, hidden, fc1].
            weight2: Unsharded native FC2 matrices, [local_experts, ffn, hidden].
            hidden_states: Dispatched tokens before expert computation.
            probs: Dispatched routing probabilities in expert order.
        """
        num_local_experts = weight1.shape[0]
        fc1_elements = weight1.shape[1] * weight1.shape[2]
        selected1 = weight1.index_select(0, self.send_indices).flatten(1)
        selected2 = weight2.index_select(0, self.send_indices).flatten(1)
        packet = torch.cat((selected1, selected2), dim=1)
        received, hidden_states, probs = _ExchangeExpertWeights.apply(
            packet, hidden_states, probs, self.group,
            self.receive_splits, self.send_splits,
        )
        # Even an empty received packet must stay connected to the weight
        # collective so owners with only outgoing replicas participate backward.
        replicas = received.new_zeros((num_local_experts, packet.shape[1])).index_copy(
            0, self.receive_slots, received,
        )
        replica1 = replicas[:, :fc1_elements].reshape_as(weight1)
        replica2 = replicas[:, fc1_elements:].reshape_as(weight2)
        runtime_weight1 = torch.cat((weight1, replica1), dim=0)
        runtime_weight2 = torch.cat((weight2, replica2), dim=0)
        return runtime_weight1, runtime_weight2, hidden_states, probs
