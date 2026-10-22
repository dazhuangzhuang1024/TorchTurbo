"""Native tensor parallel styles and model-plan preparation."""

from .expert import GroupedMLPParallel
from .parallelize import (
    PrepareModuleMetadata,
    SharedHeadNormParallel,
    apply_tensor_parallel_plan,
    register_tensor_parallel_cache_guard,
    resolve_tensor_parallel_plan,
    validate_tensor_parallel_targets,
)

__all__ = [
    "GroupedMLPParallel",
    "PrepareModuleMetadata",
    "SharedHeadNormParallel",
    "apply_tensor_parallel_plan",
    "register_tensor_parallel_cache_guard",
    "resolve_tensor_parallel_plan",
    "validate_tensor_parallel_targets",
]
