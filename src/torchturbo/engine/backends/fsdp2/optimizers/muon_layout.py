"""Muon storage-to-compute layout planning for one-dimensional FSDP."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field

import torch
from torch.distributed.tensor import Shard

from ....optimizers.muon import MatrixRegion, ResolvedMatrixView


@dataclass(frozen=True, slots=True)
class _RowPartition:
    """One group rank's contiguous rows in global storage order."""

    group_rank: int
    global_row_offset: int
    num_rows: int
    tensor_shape: torch.Size

    @property
    def global_row_end(self) -> int:
        return self.global_row_offset + self.num_rows

    @property
    def numel(self) -> int:
        return math.prod(self.tensor_shape)


@dataclass(frozen=True, slots=True)
class _MuonParameterLayout:
    """Persistent storage and logical matrices before owner assignment."""

    fqn: str
    global_storage_shape: torch.Size
    storage_partitions: tuple[_RowPartition, ...]
    matrices: tuple[MatrixRegion, ...]
    is_batched: bool


@dataclass(frozen=True, slots=True)
class _MatrixComputeGroup:
    """Same-shaped independent matrices in a contiguous compute-buffer span."""

    flat_offset: int
    tensor_shape: torch.Size


@dataclass(frozen=True, slots=True)
class _MatrixPartition:
    """Complete matrices packed by shape, then matrix index, on one rank."""

    group_rank: int
    matrix_indices: tuple[int, ...]
    tensor_shape: torch.Size
    matrix_groups: tuple[_MatrixComputeGroup, ...] = ()

    @property
    def numel(self) -> int:
        return math.prod(self.tensor_shape)


@dataclass(frozen=True, slots=True)
class _MatrixRoute:
    """A rectangular matrix region transferred between two flat tensors."""

    matrix_index: int
    global_row_offset: int
    global_column_offset: int
    num_rows: int
    num_columns: int
    source_group_rank: int
    destination_group_rank: int
    source_flat_offset: int
    destination_flat_offset: int
    source_row_stride: int
    destination_row_stride: int

    @property
    def numel(self) -> int:
        return self.num_rows * self.num_columns

    def inverse(self) -> _MatrixRoute:
        """Return the matching compute-to-storage route."""
        return _MatrixRoute(
            matrix_index=self.matrix_index,
            global_row_offset=self.global_row_offset,
            global_column_offset=self.global_column_offset,
            num_rows=self.num_rows,
            num_columns=self.num_columns,
            source_group_rank=self.destination_group_rank,
            destination_group_rank=self.source_group_rank,
            source_flat_offset=self.destination_flat_offset,
            destination_flat_offset=self.source_flat_offset,
            source_row_stride=self.destination_row_stride,
            destination_row_stride=self.source_row_stride,
        )


@dataclass(frozen=True, slots=True)
class _StorageToComputePlan:
    """One parameter's storage and temporary compute partitions."""

    storage_partitions: tuple[_RowPartition, ...]
    compute_partitions: tuple[_MatrixPartition, ...]
    matrices: tuple[MatrixRegion, ...]
    storage_to_compute_routes: tuple[_MatrixRoute, ...]
    compute_to_storage_routes: tuple[_MatrixRoute, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compute_to_storage_routes",
            tuple(route.inverse() for route in self.storage_to_compute_routes),
        )

    @property
    def requires_redistribution(self) -> bool:
        return bool(self.storage_to_compute_routes)

    @property
    def has_remote_transfers(self) -> bool:
        return any(
            route.source_group_rank != route.destination_group_rank
            for route in self.storage_to_compute_routes
        )

    def storage_partition(self, group_rank: int) -> _RowPartition:
        """Return one group rank's persistent storage partition."""
        return self.storage_partitions[group_rank]

    def compute_partition(self, group_rank: int) -> _MatrixPartition:
        """Return one group rank's temporary compute partition."""
        return self.compute_partitions[group_rank]


class _MuonLayoutPlanner:
    """Resolve persistent FSDP storage into complete Muon matrices."""

    def __init__(
        self,
        mesh_indices_in_group_order: tuple[int, ...],
        current_group_rank: int,
    ) -> None:
        num_ranks = len(mesh_indices_in_group_order)
        if sorted(mesh_indices_in_group_order) != list(range(num_ranks)):
            raise ValueError("mesh indices must be a permutation of group ranks")
        if current_group_rank < 0 or current_group_rank >= num_ranks:
            raise ValueError("current group rank is outside the communication group")
        self._mesh_indices_in_group_order = mesh_indices_in_group_order
        self._current_group_rank = current_group_rank

    def resolve_parameter(
        self,
        *,
        fqn: str,
        global_storage_shape: torch.Size,
        local_storage_shape: torch.Size,
        compute_view: ResolvedMatrixView | None = None,
    ) -> _MuonParameterLayout:
        """Resolve storage and independent matrices without choosing owners."""
        if len(global_storage_shape) != 2 or any(
            size <= 0 for size in global_storage_shape
        ):
            raise ValueError(f"Muon parameter {fqn!r} requires non-empty 2D storage")

        storage = self._build_storage_partitions(global_storage_shape)
        self._validate_storage(
            fqn, global_storage_shape, local_storage_shape, storage
        )
        if compute_view is None:
            matrices = (MatrixRegion(0, 0, 0, *global_storage_shape),)
        else:
            if compute_view.storage_shape != global_storage_shape:
                raise ValueError(
                    f"Muon parameter {fqn!r} compute view does not match its storage"
                )
            matrices = compute_view.matrices
        return _MuonParameterLayout(
            fqn=fqn,
            global_storage_shape=global_storage_shape,
            storage_partitions=storage,
            matrices=matrices,
            is_batched=compute_view is not None,
        )

    def build_plan(
        self,
        layout: _MuonParameterLayout,
        owner_group_ranks: tuple[int, ...],
    ) -> _StorageToComputePlan:
        """Build transfers for one owner group rank per logical matrix."""
        num_ranks = len(self._mesh_indices_in_group_order)
        if len(owner_group_ranks) != len(layout.matrices):
            raise ValueError(
                f"Muon parameter {layout.fqn!r} requires one owner per matrix"
            )
        if any(not 0 <= rank < num_ranks for rank in owner_group_ranks):
            raise ValueError(
                f"Muon parameter {layout.fqn!r} owner rank is outside its group"
            )

        matrix_indices_by_rank: list[list[int]] = [[] for _ in range(num_ranks)]
        for matrix_index, group_rank in enumerate(owner_group_ranks):
            matrix_indices_by_rank[group_rank].append(matrix_index)
        matrix_rows, matrix_columns = layout.matrices[0].tensor_shape
        partitions = []
        for group_rank, matrix_indices in enumerate(matrix_indices_by_rank):
            indices_by_shape: dict[torch.Size, list[int]] = {}
            for index in matrix_indices:
                shape = layout.matrices[index].tensor_shape
                indices_by_shape.setdefault(shape, []).append(index)
            ordered_indices = []
            groups = []
            offset = 0
            for shape, indices in indices_by_shape.items():
                group_shape = (
                    torch.Size((len(indices), *shape)) if layout.is_batched else shape
                )
                groups.append(_MatrixComputeGroup(offset, group_shape))
                ordered_indices.extend(indices)
                offset += group_shape.numel()
            if len(groups) == 1:
                tensor_shape = groups[0].tensor_shape
            elif groups:
                tensor_shape = torch.Size((offset,))
            else:
                tensor_shape = torch.Size(
                    (0, matrix_rows, matrix_columns)
                    if layout.is_batched else (0, matrix_columns)
                )
            partitions.append(
                _MatrixPartition(
                    group_rank, tuple(ordered_indices), tensor_shape, tuple(groups)
                )
            )
        compute = tuple(partitions)
        routes = self._build_matrix_routes(layout, compute, owner_group_ranks)
        self._validate_routes(layout, compute, routes)
        # A shape-only view needs no transfer when every element stays at its
        # original flat offset. Single-row regions do not depend on row stride.
        if all(
            route.source_group_rank == route.destination_group_rank
            and route.source_flat_offset == route.destination_flat_offset
            and (
                route.num_rows == 1
                or route.source_row_stride == route.destination_row_stride
            )
            for route in routes
        ):
            routes = ()
        return _StorageToComputePlan(
            storage_partitions=layout.storage_partitions,
            compute_partitions=compute,
            matrices=layout.matrices,
            storage_to_compute_routes=routes,
        )

    def _build_storage_partitions(
        self,
        global_shape: torch.Size,
    ) -> tuple[_RowPartition, ...]:
        global_rows, columns = global_shape
        num_ranks = len(self._mesh_indices_in_group_order)
        partitions = []
        for group_rank, mesh_index in enumerate(self._mesh_indices_in_group_order):
            local_rows, row_offset = Shard.local_shard_size_and_offset(
                global_rows, num_ranks, mesh_index
            )
            partitions.append(
                _RowPartition(
                    group_rank=group_rank,
                    global_row_offset=row_offset,
                    num_rows=local_rows,
                    tensor_shape=torch.Size((local_rows, columns)),
                )
            )
        return tuple(partitions)

    @staticmethod
    def _build_matrix_routes(
        layout: _MuonParameterLayout,
        compute: tuple[_MatrixPartition, ...],
        owner_group_ranks: tuple[int, ...],
    ) -> tuple[_MatrixRoute, ...]:
        sources = sorted(
            (
                partition
                for partition in layout.storage_partitions
                if partition.num_rows
            ),
            key=lambda partition: partition.global_row_offset,
        )
        source_row_ends = tuple(source.global_row_end for source in sources)
        matrix_flat_offsets = [0] * len(layout.matrices)
        for partition in compute:
            offset = 0
            for matrix_index in partition.matrix_indices:
                matrix_flat_offsets[matrix_index] = offset
                offset += layout.matrices[matrix_index].numel

        storage_columns = layout.global_storage_shape[1]
        routes = []
        for matrix in layout.matrices:
            matrix_row_end = matrix.global_row_offset + matrix.num_rows
            source_index = bisect_right(source_row_ends, matrix.global_row_offset)
            while source_index < len(sources):
                source = sources[source_index]
                if source.global_row_offset >= matrix_row_end:
                    break
                row_start = max(source.global_row_offset, matrix.global_row_offset)
                row_end = min(source.global_row_end, matrix_row_end)
                routes.append(
                    _MatrixRoute(
                        matrix_index=matrix.matrix_index,
                        global_row_offset=row_start,
                        global_column_offset=matrix.global_column_offset,
                        num_rows=row_end - row_start,
                        num_columns=matrix.num_columns,
                        source_group_rank=source.group_rank,
                        destination_group_rank=owner_group_ranks[matrix.matrix_index],
                        source_flat_offset=(row_start - source.global_row_offset)
                        * storage_columns
                        + matrix.global_column_offset,
                        destination_flat_offset=matrix_flat_offsets[matrix.matrix_index]
                        + (row_start - matrix.global_row_offset) * matrix.num_columns,
                        source_row_stride=storage_columns,
                        destination_row_stride=matrix.num_columns,
                    )
                )
                source_index += 1
        return tuple(routes)

    def _validate_storage(
        self,
        fqn: str,
        global_shape: torch.Size,
        local_storage_shape: torch.Size,
        storage: tuple[_RowPartition, ...],
    ) -> None:
        global_rows, columns = global_shape
        if tuple(part.group_rank for part in storage) != tuple(
            range(len(self._mesh_indices_in_group_order))
        ):
            raise RuntimeError(
                f"invalid Muon redistribution plan: {fqn!r} storage "
                "partitions do not follow group-rank order"
            )
        cursor = 0
        for partition in sorted(storage, key=lambda part: part.global_row_offset):
            if (
                partition.global_row_offset != cursor
                or partition.num_rows < 0
                or partition.tensor_shape != torch.Size((partition.num_rows, columns))
            ):
                raise RuntimeError(
                    f"invalid Muon redistribution plan: {fqn!r} storage "
                    "partitions do not exactly cover global rows"
                )
            cursor = partition.global_row_end
        if cursor != global_rows:
            raise RuntimeError(
                f"invalid Muon redistribution plan: {fqn!r} storage "
                "partitions do not exactly cover global rows"
            )
        predicted_local_shape = storage[self._current_group_rank].tensor_shape
        if predicted_local_shape != local_storage_shape:
            raise RuntimeError(
                f"Muon parameter {fqn!r} logical local shard does not match its mesh"
            )

    @staticmethod
    def _validate_routes(
        layout: _MuonParameterLayout,
        compute: tuple[_MatrixPartition, ...],
        routes: tuple[_MatrixRoute, ...],
    ) -> None:
        storage = layout.storage_partitions
        matrices = layout.matrices
        num_ranks = len(storage)
        if tuple(part.group_rank for part in compute) != tuple(range(num_ranks)):
            raise RuntimeError(
                f"invalid Muon redistribution plan: {layout.fqn!r} compute "
                "partitions do not follow group-rank order"
            )
        matrix_positions: dict[int, tuple[int, int]] = {}
        for partition in compute:
            offset = 0
            for matrix_index in partition.matrix_indices:
                if (
                    not 0 <= matrix_index < len(matrices)
                    or matrix_index in matrix_positions
                ):
                    raise RuntimeError(
                        f"invalid Muon redistribution plan: {layout.fqn!r} "
                        "compute partitions do not exactly cover matrices"
                    )
                matrix_positions[matrix_index] = (partition.group_rank, offset)
                offset += matrices[matrix_index].numel
            if offset != partition.numel:
                raise RuntimeError(
                    f"invalid Muon redistribution plan: {layout.fqn!r} "
                    "compute partition has an invalid tensor shape"
                )
        if len(matrix_positions) != len(matrices):
            raise RuntimeError(
                f"invalid Muon redistribution plan: {layout.fqn!r} "
                "compute partitions do not exactly cover matrices"
            )

        row_spans_by_matrix: list[list[tuple[int, int]]] = [[] for _ in matrices]
        storage_columns = layout.global_storage_shape[1]
        for route in routes:
            if (
                not 0 <= route.matrix_index < len(matrices)
                or not 0 <= route.source_group_rank < num_ranks
                or not 0 <= route.destination_group_rank < num_ranks
            ):
                raise RuntimeError(
                    f"invalid Muon redistribution plan: {layout.fqn!r} "
                    "has an invalid route"
                )
            matrix = matrices[route.matrix_index]
            source = storage[route.source_group_rank]
            destination = compute[route.destination_group_rank]
            owner_rank, matrix_flat_offset = matrix_positions[route.matrix_index]
            matrix_row_offset = route.global_row_offset - matrix.global_row_offset
            source_row_offset = route.global_row_offset - source.global_row_offset
            if (
                route.num_rows <= 0
                or matrix_row_offset < 0
                or matrix_row_offset + route.num_rows > matrix.num_rows
                or source_row_offset < 0
                or source_row_offset + route.num_rows > source.num_rows
                or route.global_column_offset != matrix.global_column_offset
                or route.num_columns != matrix.num_columns
                or route.destination_group_rank != owner_rank
                or route.source_row_stride != storage_columns
                or route.destination_row_stride != matrix.num_columns
                or route.source_flat_offset
                != source_row_offset * storage_columns + matrix.global_column_offset
                or route.destination_flat_offset
                != matrix_flat_offset + matrix_row_offset * matrix.num_columns
                or route.source_flat_offset < 0
                or route.destination_flat_offset < 0
                or route.source_flat_offset
                + (route.num_rows - 1) * route.source_row_stride
                + route.num_columns
                > source.numel
                or route.destination_flat_offset
                + (route.num_rows - 1) * route.destination_row_stride
                + route.num_columns
                > destination.numel
            ):
                raise RuntimeError(
                    f"invalid Muon redistribution plan: {layout.fqn!r} "
                    "has an invalid route"
                )
            row_spans_by_matrix[route.matrix_index].append(
                (matrix_row_offset, route.num_rows)
            )

        # The resolved matrices tile storage, and each has one packed compute
        # location. Exact row coverage per matrix therefore proves both tensor
        # coverages without expanding column-group rectangles into row routes.
        for matrix, spans in zip(matrices, row_spans_by_matrix, strict=True):
            cursor = 0
            for offset, num_rows in sorted(spans):
                if offset != cursor:
                    raise RuntimeError(
                        f"invalid Muon redistribution plan: {layout.fqn!r} matrix "
                        f"{matrix.matrix_index} routes do not exactly cover its rows"
                    )
                cursor += num_rows
            if cursor != matrix.num_rows:
                raise RuntimeError(
                    f"invalid Muon redistribution plan: {layout.fqn!r} matrix "
                    f"{matrix.matrix_index} routes do not exactly cover its rows"
                )
