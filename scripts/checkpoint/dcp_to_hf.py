#!/usr/bin/env python
"""Export a TorchTurbo training checkpoint as HF safetensors.

Rebuilds the model using the checkpoint's reconstruction metadata, restores
native model state, and writes ``<output_dir>/huggingface``. Keep export output
separate from training checkpoints so checkpoint rotation is unaffected.

Usage:
    torchrun --nproc_per_node=8 scripts/checkpoint/dcp_to_hf.py \
        --checkpoint_dir output/my_run/checkpoint-1000 \
        --output_dir exports/my_run/step-1000
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torchturbo.engine import EngineConfig, TorchTurboArgs


# Ensure torchturbo is importable when running without pip install -e
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_REPO_SRC = os.path.join(_REPO_ROOT, "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the export arguments left after removing ``--tt_*`` flags."""
    parser = argparse.ArgumentParser(
        description="Export a TorchTurbo training checkpoint as HF safetensors."
    )
    parser.add_argument(
        "--model_name_or_path",
        help="Override the saved source model path, for example after relocation.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Training checkpoint to export, e.g. output/checkpoint-1000.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="The export is written to <output_dir>/huggingface.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank of the training run; omit for full fine-tuning.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha of the training run; required for legacy LoRA checkpoints.",
    )
    parser.add_argument(
        "--lora_target",
        default=None,
        help="Comma-separated LoRA target modules, or all-linear.",
    )
    parser.add_argument("--use_rslora", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_dora", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--adapter_config", help="PEFT adapter_config.json for a legacy checkpoint.")
    return parser.parse_args(argv)


def build_lora_config(
    args: argparse.Namespace, metadata: dict[str, Any] | None = None
) -> Any | None:
    """Read saved PEFT settings, rejecting conflicting explicit options."""
    config = None
    if metadata is not None:
        configs = metadata["peft_configs"]
        if configs:
            from peft import PeftConfig

            if list(configs) != ["default"]:
                raise ValueError("Export currently requires a single default PEFT adapter.")
            # Like PeftConfig.from_pretrained, ignore fields from a newer PEFT.
            config = PeftConfig.from_peft_type(**configs["default"])
    if args.adapter_config is not None:
        import json
        from peft import PeftConfig

        with open(args.adapter_config) as file:
            explicit_config = json.load(file)
        if metadata is not None:
            raise ValueError("Use the checkpoint's saved adapter configuration for export.")
        config = PeftConfig.from_peft_type(**explicit_config)

    targets = args.lora_target
    if targets is not None:
        targets = "all-linear" if targets in ("all", "all-linear") else set(
            module.strip() for module in targets.split(",") if module.strip()
        )
    overrides = {
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": targets,
        "use_rslora": args.use_rslora,
        "use_dora": args.use_dora,
    }
    if config is not None:
        values = config.to_dict()
        for key, value in overrides.items():
            if value is not None and value != values[key]:
                raise ValueError(f"Explicit {key}={value!r} differs from saved {values[key]!r}.")
        return config
    if metadata is not None:
        if any(value is not None for value in overrides.values()):
            raise ValueError("The checkpoint was saved without PEFT adapters.")
        return None
    if args.lora_rank is None:
        if any(value is not None for value in overrides.values()):
            raise ValueError("Legacy LoRA export requires --lora_rank or --adapter_config.")
        return None
    if any(value is None for value in overrides.values()):
        raise ValueError(
            "Legacy LoRA export requires --adapter_config or explicit rank, alpha, "
            "targets, --use_rslora/--no-use_rslora and --use_dora/--no-use_dora."
        )
    from peft import LoraConfig, TaskType

    return LoraConfig(
        **overrides, lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM
    )


def validate_export_paths(checkpoint_dir: str, output_dir: str) -> None:
    """Keep standalone exports outside training checkpoint trees.

    Exporting replaces the model components of ``output_dir`` and removes the
    resume state of a training checkpoint there.
    """
    from torchturbo.engine.backends.fsdp2.checkpoint.layout import (
        MODEL_DCP_DIR,
        OPTIMIZER_DCP_DIR,
    )
    from torchturbo.engine.checkpoint.layout import SCHEDULER_FILE

    checkpoint = os.path.realpath(checkpoint_dir)
    output = os.path.realpath(output_dir)
    if os.path.commonpath((checkpoint, output)) in (checkpoint, output):
        raise ValueError("Export output must be separate from the source checkpoint directory.")
    # trainer_state.json is the HF Trainer state saved in every checkpoint.
    resume_state = (
        MODEL_DCP_DIR, OPTIMIZER_DCP_DIR, SCHEDULER_FILE, "trainer_state.json"
    )
    found = [
        name for name in resume_state if os.path.exists(os.path.join(output, name))
    ]
    if found:
        raise ValueError(
            f"Export output {output_dir} contains training checkpoint state "
            f"({', '.join(found)}); exporting there would replace its model and "
            "delete its resume state. Use a new directory or a previous export "
            "directory."
        )


def build_export_config(
    tt_args: TorchTurboArgs, metadata: dict[str, Any] | None
) -> EngineConfig:
    """Seed structural settings from metadata, retaining ordinary runtime overrides."""
    from torchturbo.engine import build_from_args

    if metadata is None:
        return build_from_args(tt_args)
    saved = metadata["engine_config"]
    if tt_args.backend is None:
        tt_args.backend = saved["backend"]
    if tt_args.mixed_precision is None:
        tt_args.mixed_precision = saved["mixed_precision"]
    degrees = ("ep_size",)
    if metadata["schema_version"] == 2:
        degrees = ("sp_size", "tp_size", "ep_size", "etp_size")
    for degree in degrees:
        # Args are the base layer; explicit CLI, environment and file values
        # still take precedence and are checked after config reconstruction.
        if degree == "sp_size":
            tt_args.sp_size = saved["parallel_plan"][degree]
        elif degree == "tp_size":
            tt_args.tp_size = saved["parallel_plan"][degree]
        elif degree == "ep_size":
            tt_args.ep_size = saved["parallel_plan"][degree]
        else:
            tt_args.etp_size = saved["parallel_plan"][degree]
    tt_args.enable_mtp = saved["mtp"]["enabled"]
    config = build_from_args(tt_args)
    current_degrees = {
        "sp_size": config.parallel_plan.sp_size, "tp_size": config.parallel_plan.tp_size,
        "ep_size": config.parallel_plan.ep_size, "etp_size": config.parallel_plan.etp_size,
    }
    for degree in degrees:
        if current_degrees[degree] != saved["parallel_plan"][degree]:
            label = degree.removesuffix("_size").upper()
            raise ValueError(f"Export {label} size must match the saved checkpoint.")
    if config.mtp.enabled != saved["mtp"]["enabled"]:
        raise ValueError("Export MTP setting must match the saved checkpoint.")
    return config


def main(argv: list[str] | None = None) -> None:
    import torch.distributed as dist

    from torchturbo.engine import (
        CheckpointFormat,
        TorchTurboEngine,
        init_process_group_if_needed,
    )
    from torchturbo.engine.args import parse_torchturbo_args, set_torchturbo_args
    from torchturbo.engine.checkpoint.distributed import rank_zero_call
    from torchturbo.engine.checkpoint.layout import validate_output_dir
    from torchturbo.engine.checkpoint.metadata import load_metadata

    tt_args, remaining_argv = parse_torchturbo_args(
        sys.argv[1:] if argv is None else argv
    )
    set_torchturbo_args(tt_args)
    args = parse_args(remaining_argv)
    validate_export_paths(args.checkpoint_dir, args.output_dir)

    # The CPU backend serves full CPU offload, if the --tt_* settings enable it.
    init_process_group_if_needed(enable_cpu_backend=True)
    try:
        metadata = rank_zero_call(lambda: load_metadata(args.checkpoint_dir))
        if metadata is not None and metadata["schema_version"] == 2:
            saved_world = metadata["parallel_layout"]["topology"]["world_size"]
            if dist.get_world_size() != saved_world:
                raise ValueError(f"Export WORLD must equal saved WORLD ({saved_world}).")
            saved_layout = metadata["parallel_layout"]
            if (
                saved_layout["layout_schema_version"] != 1
                or saved_layout["expert_layout_version"] not in ("packed_2d_v1", "gated_logical_v1")
            ):
                raise ValueError("Export requires a supported TP checkpoint layout.")
            topology = saved_layout["topology"]
            sp, tp, ep, etp = (topology[name] for name in ("sp_size", "tp_size", "ep_size", "etp_size"))
            if (
                min(sp, tp, ep, etp) < 1
                or saved_world % (sp * tp) != 0
                or saved_world % (ep * etp) != 0
                or (tp == 1 and etp != 1)
                or (saved_layout["expert_layout_version"] == "gated_logical_v1"
                    and not (tp == etp and etp > 1))
                or (saved_layout["expert_layout_version"] == "packed_2d_v1"
                    and tp > 1 and (ep != 1 or etp != 1))
            ):
                raise ValueError(
                    "Export requires a supported topology with aligned TP=ETP for logical "
                    "experts, EP=ETP=1 for packed layouts under TP, and ETP=1 when TP=1."
                )
            if metadata["engine_config"]["custom_tensor_parallel_plan"]:
                raise ValueError(
                    "Export requires the custom Python TP plan to be provided again "
                    "by a caller. The standalone CLI cannot reconstruct that plan."
                )
        source_path = args.model_name_or_path
        if source_path is None and metadata is not None:
            source_path = metadata["source_path"]
        if source_path is None:
            raise ValueError("Legacy checkpoints require --model_name_or_path.")
        validate_output_dir(args.output_dir, source_path)
        config = build_export_config(tt_args, metadata)
        peft_config = build_lora_config(args, metadata)
        init_kwargs = {}
        if metadata is not None:
            from transformers import AutoConfig

            init_kwargs["config"] = AutoConfig.from_pretrained(
                source_path,
                trust_remote_code=args.trust_remote_code,
                **metadata["model_config"],
            )
        engine = TorchTurboEngine(config)
        model = engine.prepare_model(
            model_name_or_path=source_path,
            trust_remote_code=args.trust_remote_code,
            peft_config=peft_config,
            init_kwargs=init_kwargs,
        )
        engine.load_from_checkpoint(model, args.checkpoint_dir)
        engine.save_model(model, args.output_dir, format=CheckpointFormat.HF)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
