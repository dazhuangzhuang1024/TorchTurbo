"""Gradient norms and optimizer groups for native TP parameter layouts."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.nn.utils import clip_grads_with_norm_, get_total_norm

from .parallel_context import ParameterOwnership
from ...distributed import assert_finite


def _parameter_buckets(
    parameters: Iterable[nn.Parameter], ownership: ParameterOwnership,
) -> dict[tuple, list[nn.Parameter]]:
    buckets: dict[tuple, list[nn.Parameter]] = {}
    for parameter in parameters:
        mesh = parameter.device_mesh if isinstance(parameter, DTensor) else None
        placements = parameter.placements if isinstance(parameter, DTensor) else None
        key = (ownership.is_expert(parameter), mesh, placements)
        buckets.setdefault(key, []).append(parameter)
    return buckets


def split_optimizer_groups_by_layout(
    param_groups: Sequence[Mapping[str, Any]], ownership: ParameterOwnership,
) -> list[dict[str, Any]]:
    """Split each optimizer group into ordered ownership and layout buckets."""
    result = []
    for group in param_groups:
        options = {key: value for key, value in group.items() if key != "params"}
        trainable = (parameter for parameter in group["params"] if parameter.requires_grad)
        for parameters in _parameter_buckets(trainable, ownership).values():
            result.append({"params": parameters, **options})
    return result


@torch.no_grad()
def clip_grad_norm_by_layout_(
    parameters: Iterable[nn.Parameter],
    max_norm: float,
    norm_type: float,
    *,
    ownership: ParameterOwnership,
    ep_group: dist.ProcessGroup | None,
    device: torch.device,
) -> torch.Tensor:
    """Clip TP gradients with a norm counting each logical parameter once.

    Args:
        parameters: Parameters with rank-consistent gradient participation.
        max_norm: Upper bound for the global norm.
        norm_type: Positive finite norm order or positive infinity.
        ownership: Persistent backend registry identifying EP-owned parameters.
        ep_group: Group connecting distinct expert owners on the same ETP lane.
        device: Device used to combine scalar norms and check finiteness.

    Returns:
        The plain global gradient norm before clipping.
    """
    norm_type = float(norm_type)
    if not norm_type > 0:
        raise ValueError("TP gradient norm_type must be positive or positive infinity")

    buckets = _parameter_buckets(
        (parameter for parameter in parameters if parameter.grad is not None), ownership,
    )
    dense_norms, expert_norms, local_norms = [], [], []
    for (is_expert, mesh, placements), bucket in buckets.items():
        gradients = [parameter.grad for parameter in bucket]
        if norm_type == float("inf"):
            # Native max norms have no identity for empty FSDP shards. Reduce
            # local maxima only over dimensions that own distinct shard data.
            local_gradients = [
                gradient.to_local() if isinstance(gradient, DTensor) else gradient
                for gradient in gradients
            ]
            nonempty = [gradient for gradient in local_gradients if gradient.numel()]
            if nonempty:
                norm = get_total_norm(nonempty, norm_type).to(device)
            else:
                norm = torch.zeros((), device=device, dtype=gradients[0].dtype)
            # MAX may hide NaNs; retain the local evidence for the WORLD check.
            local_norms.append(norm.clone())
            if mesh is not None:
                for dim, placement in enumerate(placements):
                    if isinstance(placement, (Shard, _StridedShard)):
                        dist.all_reduce(
                            norm, op=dist.ReduceOp.MAX, group=mesh.get_group(dim)
                        )
        else:
            norm = get_total_norm(gradients, norm_type)
            if isinstance(norm, DTensor):
                norm = norm.full_tensor()
            norm = norm.to(device)
        (expert_norms if is_expert else dense_norms).append(norm)

    if expert_norms:
        owner_norm = get_total_norm(expert_norms, norm_type)
        owner_norms = [
            torch.empty_like(owner_norm) for _ in range(dist.get_world_size(ep_group))
        ]
        dist.all_gather(owner_norms, owner_norm, group=ep_group)
        dense_norms.extend(owner_norms)

    if dense_norms:
        total_norm = get_total_norm(dense_norms, norm_type).to(device)
    else:
        total_norm = torch.zeros((), device=device)
    assert_finite(torch.stack(local_norms + [total_norm]), "gradient norm")
    for bucket in buckets.values():
        clip_grads_with_norm_(bucket, max_norm, total_norm)
    return total_norm
