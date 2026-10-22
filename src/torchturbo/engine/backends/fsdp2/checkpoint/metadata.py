"""Resolved parameter layouts for native TP checkpoint compatibility."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from torchturbo.engine.backends.fsdp2.parallel_context import (
    ParallelContext,
    ParameterOwnership,
)


def build_parallel_layout(
    model: nn.Module, context: ParallelContext, ownership: ParameterOwnership
) -> dict[str, Any]:
    """Gather actual prepared layouts across every WORLD topology lane.

    Native placement strings retain dimensions and strided-shard split factors.
    Dtypes are omitted to preserve supported mixed-precision loading.
    """
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    local = []
    layout_table: dict[str, int] = {}
    layout_ids = []
    expert_fqns = ownership.expert_fqns(model)
    for name, parameter in model.named_parameters():
        expert = name in expert_fqns
        layout = {"mesh": None, "mesh_dim_names": [], "placements": []}
        if isinstance(parameter, DTensor):
            layout = {
                "mesh": parameter.device_mesh.mesh.tolist(),
                "mesh_dim_names": list(parameter.device_mesh.mesh_dim_names),
                "placements": [str(placement) for placement in parameter.placements],
            }
        if expert:
            ep_rank = context.get_ep_info()["ep_rank"]
            layout["ep_rank"] = ep_rank
            if context.etp_size > 1:
                experts = model.get_submodule(name.rsplit(".", 1)[0])
                start = ep_rank * experts.num_local_experts
                layout["expert_ids"] = list(range(start, start + experts.num_local_experts))
        descriptor = json.dumps(layout, sort_keys=True)
        if descriptor not in layout_table:
            layout_table[descriptor] = len(layout_table)
        layout_ids.append(layout_table[descriptor])
        local.append({
            "fqn": name, "aliases": sorted(aliases[id(parameter)]),
            "shape": list(parameter.shape), "ownership": "expert" if expert else "dense",
        })
    descriptors = list(layout_table)
    fingerprints = [hashlib.sha256(descriptor.encode()).digest() for descriptor in descriptors]
    local_fqns = [entry["fqn"] for entry in local]
    lanes: list[Any] = [None] * context.root_mesh.size()
    dist.all_gather_object(lanes, (local_fqns, layout_ids, fingerprints))
    for lane_index, (lane_fqns, _, _) in enumerate(lanes):
        if lane_fqns != local_fqns:
            raise ValueError(
                "Checkpoint parameter names must match in order and count across "
                f"WORLD lanes; lane {lane_index} differs from the local model."
            )

    # Full mesh lists travel once per distinct layout, without WORLD-sized
    # all_gather_object padding or one copy per parameter on every rank.
    owners: dict[bytes, tuple[int, int]] = {}
    for rank, (_, _, lane_fingerprints) in enumerate(lanes):
        for index, fingerprint in enumerate(lane_fingerprints):
            owners.setdefault(fingerprint, (rank, index))
    gathered_descriptors = {}
    local_rank = dist.get_rank()
    for fingerprint in sorted(owners):
        rank, index = owners[fingerprint]
        value = [descriptors[index] if rank == local_rank else None]
        dist.broadcast_object_list(value, src=rank)
        gathered_descriptors[fingerprint] = value[0]

    parameters = []
    for index, entry in enumerate(local):
        layouts = set()
        for _, lane_layout_ids, lane_fingerprints in lanes:
            fingerprint = lane_fingerprints[lane_layout_ids[index]]
            layouts.add(gathered_descriptors[fingerprint])
        parameters.append({**entry, "layouts": [json.loads(item) for item in sorted(layouts)]})
    return {
        "layout_schema_version": 1,
        "expert_layout_version": (
            "gated_logical_v1" if expert_fqns and context.etp_size > 1 else "packed_2d_v1"
        ),
        "topology": {
            "world_size": context.root_mesh.size(), "sp_size": context.sp_size,
            "tp_size": context.tp_size, "ep_size": context.ep_size,
            "etp_size": context.etp_size,
        },
        "parameters": sorted(parameters, key=lambda entry: entry["fqn"]),
    }
