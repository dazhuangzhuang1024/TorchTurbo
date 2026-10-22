"""Shared mesh views and stable parameter ownership for distributed backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


def validate_parallel_degrees(
    world_size: int, *, sp_size: int, ep_size: int, tp_size: int, etp_size: int,
) -> None:
    """Require positive integer degrees and mesh products dividing WORLD."""
    degrees = (("sp_size", sp_size), ("ep_size", ep_size),
               ("tp_size", tp_size), ("etp_size", etp_size))
    for name, size in degrees:
        if type(size) is not int or size < 1 or world_size % size:
            raise ValueError(
                f"{name} ({size}) must be a positive integer and divide "
                f"world size ({world_size})."
            )
    for name, size in (("sp_size * tp_size", sp_size * tp_size),
                       ("ep_size * etp_size", ep_size * etp_size)):
        if world_size % size:
            raise ValueError(f"{name} ({size}) must divide world size ({world_size}).")


class ParallelContext:
    """Derive dense, expert, and data process groups from one WORLD mesh.

    Args:
        device_type: Device type used by the process groups.
        sp_size: Number of consecutive ranks processing one sequence.
        ep_size: Number of ranks holding different experts on each ETP lane.
        tp_size: Number of consecutive ranks splitting dense computation.
        etp_size: Number of consecutive ranks splitting expert computation.
    """

    def __init__(
        self, device_type: str, *, sp_size: int, ep_size: int,
        tp_size: int = 1, etp_size: int = 1,
    ) -> None:
        world_size = dist.get_world_size()
        validate_parallel_degrees(
            world_size, sp_size=sp_size, ep_size=ep_size,
            tp_size=tp_size, etp_size=etp_size,
        )
        self.sp_size: int = sp_size
        self.ep_size: int = ep_size
        self.tp_size: int = tp_size
        self.etp_size: int = etp_size
        self.root_mesh: DeviceMesh = init_device_mesh(
            device_type, (world_size,), mesh_dim_names=("dp",)
        )
        self.tp_mesh: DeviceMesh | None = None
        self.etp_mesh: DeviceMesh | None = None
        # Integer dimensions avoid the PyTorch 2.10 named-unflatten bug.
        if tp_size == 1:
            self.dense_mesh: DeviceMesh = self.root_mesh
            data_mesh = self.root_mesh._unflatten(
                0, (world_size // sp_size, sp_size), ("data_dp", "sp")
            )
        else:
            self.dense_mesh = self.root_mesh._unflatten(
                0, (world_size // tp_size, tp_size), ("dp", "tp")
            )
            self.tp_mesh = self.dense_mesh["tp"]
            data_mesh = self.dense_mesh._unflatten(
                0, (world_size // (sp_size * tp_size), sp_size), ("data_dp", "sp")
            )
        self.dense_fsdp_mesh: DeviceMesh = (
            self.root_mesh if tp_size == 1 else self.dense_mesh["dp"]
        )
        if etp_size == 1:
            self.expert_mesh: DeviceMesh = self.root_mesh._unflatten(
                0, (world_size // ep_size, ep_size), ("edp", "ep")
            )
        else:
            self.expert_mesh = self.root_mesh._unflatten(
                0, (world_size // (ep_size * etp_size), ep_size, etp_size),
                ("edp", "ep", "etp"),
            )
            self.etp_mesh = self.expert_mesh["etp"]
        self.data_dp_mesh: DeviceMesh = data_mesh["data_dp"]
        self.sp_mesh: DeviceMesh = data_mesh["sp"]
        self.expert_fsdp_mesh: DeviceMesh = self.expert_mesh["edp"]
        self.ep_mesh: DeviceMesh = self.expert_mesh["ep"]
        self.expert_checkpoint_owner_mesh: DeviceMesh = (
            self.expert_fsdp_mesh if etp_size == 1
            else self.expert_mesh["edp", "etp"]._flatten("expert_checkpoint_owner")
        )
        self.expert_checkpoint_owner_group: dist.ProcessGroup = (
            self.expert_checkpoint_owner_mesh.get_group()
        )
        self.expert_checkpoint_owner_ranks: list[int] = dist.get_process_group_ranks(
            self.expert_checkpoint_owner_group
        )

    def get_dp_info(self) -> dict[str, Any]:
        """Return the data sampler's rank, size, and process group."""
        return {
            "dp_rank": self.data_dp_mesh.get_local_rank(),
            "dp_size": self.data_dp_mesh.size(),
            "dp_group": (
                dist.group.WORLD
                if self.data_dp_mesh.size() == self.root_mesh.size()
                else self.data_dp_mesh.get_group()
            ),
        }

    def get_sp_info(self) -> dict[str, Any]:
        """Return the sequence-parallel rank, size, and process group."""
        return {
            "sp_rank": self.sp_mesh.get_local_rank(),
            "sp_size": self.sp_mesh.size(),
            "sp_group": self.sp_mesh.get_group(),
        }

    def get_ep_info(self) -> dict[str, Any]:
        """Return the expert-parallel rank, size, and process group."""
        return {
            "ep_rank": self.ep_mesh.get_local_rank(),
            "ep_size": self.ep_mesh.size(),
            "ep_group": self.ep_mesh.get_group(),
        }

    def get_tp_info(self) -> dict[str, Any]:
        """Return the tensor-parallel rank, size, and process group."""
        return {
            "tp_rank": self.tp_mesh.get_local_rank() if self.tp_mesh is not None else 0,
            "tp_size": self.tp_size,
            "tp_group": self.tp_mesh.get_group() if self.tp_mesh is not None else None,
        }

    def get_etp_info(self) -> dict[str, Any]:
        """Return the expert tensor-parallel rank, size, and process group."""
        return {
            "etp_rank": self.etp_mesh.get_local_rank() if self.etp_mesh is not None else 0,
            "etp_size": self.etp_size,
            "etp_group": self.etp_mesh.get_group() if self.etp_mesh is not None else None,
        }


@dataclass(frozen=True, slots=True)
class _ModelOwnership:
    expert_fqns: frozenset[str]
    parameter_ids: frozenset[int]


class ParameterOwnership:
    """Track expert ownership separately for every live prepared model.

    Register finalized FSDP parameters before using a model's names or binding
    replacements. Entries retain only names and IDs so discarded models and
    their parameters can be released.
    """

    def __init__(self) -> None:
        self._models: WeakKeyDictionary[nn.Module, _ModelOwnership] = WeakKeyDictionary()

    def register(self, model: nn.Module, expert_parameters: set[nn.Parameter]) -> None:
        """Record and bind all aliases owned by this model's expert FSDP units."""
        expert_fqns = frozenset(
            name
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter in expert_parameters
        )
        self._models[model] = _ModelOwnership(
            expert_fqns, frozenset(id(parameter) for parameter in expert_parameters)
        )

    def expert_fqns(self, model: nn.Module) -> frozenset[str]:
        """Return the registered expert names for this model only."""
        return self._models[model].expert_fqns

    def bind(self, model: nn.Module) -> None:
        """Refresh only this registered model after Parameters are replaced."""
        expert_fqns = self.expert_fqns(model)
        self._models[model] = _ModelOwnership(
            expert_fqns,
            frozenset(id(model.get_parameter(name)) for name in expert_fqns),
        )

    def is_expert(self, parameter: nn.Parameter) -> bool:
        """Return whether a parameter is an expert in any live prepared model."""
        parameter_id = id(parameter)
        return any(
            parameter_id in ownership.parameter_ids for ownership in self._models.values()
        )
