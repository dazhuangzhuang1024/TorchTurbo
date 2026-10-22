"""Versioned model reconstruction information stored alongside native state."""

from __future__ import annotations

import dataclasses
import json
import os
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from ..utils import is_peft_adapter_param
from .layout import METADATA_DIR, save_json, sync_directory

if TYPE_CHECKING:
    from ..config import EngineConfig

METADATA_FILE = "checkpoint.json"
SCHEMA_VERSION = 1
TP_SCHEMA_VERSION = 2

# PEFT config fields that may change when resuming: bookkeeping that differs
# across environments or PEFT versions, training-only settings, and
# initialization recipes whose result the loaded state overwrites. Every other
# field describes adapter structure, how its tensors are interpreted, or LoRA
# scaling. DCP loading cannot check those: it compares tensor names and shapes
# only, scaling is not stored in tensors, and saved tensors the current model
# no longer uses (for example after disabling DoRA) are ignored.
_PEFT_CHANGEABLE_FIELDS = frozenset({
    "auto_mapping", "peft_version", "base_model_name_or_path", "revision",
    "inference_mode", "lora_dropout", "bias", "init_lora_weights",
    "loftq_config", "eva_config", "corda_config", "lora_ga_config",
})


def _json_default(value: Any) -> Any:
    """Store enums as their values and sets as sorted lists.

    Sorting keeps the saved form independent of set iteration order, so a live
    PEFT config compares equal to the one saved from it.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Checkpoint configuration is not JSON serializable: {type(value)}")


def _model_peft_configs(model: nn.Module) -> dict[str, Any]:
    """Return the PEFT configs installed on ``model`` in their saved form."""
    # Models trained without PEFT adapters have no ``peft_config``.
    if not any(is_peft_adapter_param(name) for name, _ in model.named_parameters()):
        return {}
    configs = {name: config.to_dict() for name, config in model.peft_config.items()}
    return json.loads(json.dumps(configs, default=_json_default))


def build_metadata(
    model: nn.Module, config: EngineConfig, source_path: str,
    *, parallel_layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture the source, model configuration, and reconstruction settings.

    ``peft_configs`` maps each adapter name to the full PEFT config installed
    on ``model``, including its expanded ``target_modules``.
    ``peft.PeftConfig.from_peft_type`` rebuilds each config from the saved form.
    """
    if os.path.isdir(source_path):
        source_path = os.path.realpath(source_path)
    ep_size = (
        parallel_layout["topology"]["ep_size"] if parallel_layout is not None
        else config.parallel_plan.ep_size
    )
    data = {
        "schema_version": SCHEMA_VERSION,
        "source_path": source_path,
        "model_config": model.config.to_dict(),
        "engine_config": {
            "backend": config.backend.value,
            "mixed_precision": config.mixed_precision.value,
            "parallel_plan": {"ep_size": ep_size},
            "mtp": dataclasses.asdict(config.mtp),
            "freeze_policy": dataclasses.asdict(config.freeze_policy),
        },
        "peft_configs": _model_peft_configs(model),
    }
    if parallel_layout is not None:
        data["schema_version"] = TP_SCHEMA_VERSION
        data["parallel_layout"] = parallel_layout
        data["engine_config"]["parallel_plan"] = {
            name: parallel_layout["topology"][name]
            for name in ("sp_size", "tp_size", "ep_size", "etp_size")
        }
        # On the TP path the engine config retains caller overrides; model
        # defaults are reconstructed by the model class during export.
        data["engine_config"]["custom_tensor_parallel_plan"] = bool(
            config.parallel_plan is not None
            and (
                config.parallel_plan.tensor_parallel_plan.plan
                or config.parallel_plan.tensor_parallel_plan.validator is not None
            )
        )
    return json.loads(json.dumps(data, default=_json_default))


def validate_parallel_layout(
    saved: dict[str, Any] | None, current: dict[str, Any] | None
) -> None:
    """Reject native TP topology or layout migration before tensor mutation."""
    if saved is None or saved["schema_version"] == SCHEMA_VERSION:
        if current is not None:
            raise ValueError(
                "Native TP restore requires schema 2 layout metadata. Export the "
                "legacy DCP checkpoint to HF and load its weights in a new run."
            )
        return
    expected = saved["parallel_layout"]
    if current is None:
        raise ValueError("Checkpoint topology and layout require the saved TP configuration.")
    for field in ("topology", "layout_schema_version", "expert_layout_version", "parameters"):
        if expected[field] != current[field]:
            raise ValueError(
                f"Checkpoint {field} layout differs from the current prepared model. "
                "Native TP restore requires the same topology, parameter layouts, "
                "and ownership. Export to HF for weights-only migration."
            )


def save_metadata(data: dict[str, Any], checkpoint_dir: str) -> None:
    """Save reconstruction metadata on rank zero before the model component."""
    directory = os.path.join(checkpoint_dir, METADATA_DIR)
    save_json(data, os.path.join(directory, METADATA_FILE))
    sync_directory(checkpoint_dir)


def load_metadata(checkpoint_dir: str) -> dict[str, Any] | None:
    """Read reconstruction data; legacy checkpoints have no metadata directory."""
    directory = os.path.join(checkpoint_dir, METADATA_DIR)
    if not os.path.exists(directory):
        return None
    try:
        with open(os.path.join(directory, METADATA_FILE)) as file:
            data = json.load(file)
        if data["schema_version"] not in (SCHEMA_VERSION, TP_SCHEMA_VERSION):
            raise ValueError(f"Unsupported checkpoint schema {data['schema_version']!r}")
        if data["schema_version"] == TP_SCHEMA_VERSION:
            layout = data["parallel_layout"]
            for field in ("topology", "layout_schema_version", "expert_layout_version", "parameters"):
                layout[field]
            for degree in ("sp_size", "tp_size", "ep_size", "etp_size"):
                if data["engine_config"]["parallel_plan"][degree] != layout["topology"][degree]:
                    raise ValueError(
                        f"Checkpoint {degree} topology disagrees with engine_config."
                    )
        return data
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Invalid checkpoint metadata in {directory}: {exc}") from exc


def validate_peft_configs(model: nn.Module, saved: dict[str, Any]) -> None:
    """Require the adapters of this run to match the checkpoint's adapters.

    Every PEFT config field must keep its saved value, except the fields in
    ``_PEFT_CHANGEABLE_FIELDS``. Adapters present only in the checkpoint do not
    constrain this load. Fields missing from the saved config use the current
    PEFT defaults for comparison. Fields unknown to the current PEFT are skipped.

    Args:
        model: Model configured for this run.
        saved: Checkpoint metadata returned by :func:`load_metadata`.

    Raises:
        ValueError: An adapter of this run is missing from the checkpoint, or
            one of its compared fields differs from the saved value.
    """
    current = _model_peft_configs(model)
    expected = saved["peft_configs"]
    missing = sorted(current.keys() - expected.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing required PEFT adapters: {missing}.")
    for name, config in current.items():
        saved_config = dict(expected[name])
        missing_fields = config.keys() - saved_config.keys() - _PEFT_CHANGEABLE_FIELDS
        if missing_fields:
            defaults = type(model.peft_config[name])().to_dict()
            defaults = json.loads(json.dumps(defaults, default=_json_default))
            for key in missing_fields:
                saved_config[key] = defaults[key]
        compared = (config.keys() & saved_config.keys()) - _PEFT_CHANGEABLE_FIELDS
        differences = {
            key: (saved_config[key], config[key])
            for key in sorted(compared)
            if saved_config[key] != config[key]
        }
        if differences:
            raise ValueError(
                f"PEFT config of adapter {name!r} differs from the checkpoint: "
                f"{differences} (saved, current). Resume with the adapter "
                "configuration the checkpoint was trained with."
            )
