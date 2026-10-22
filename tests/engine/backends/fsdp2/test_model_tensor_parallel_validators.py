"""Model contracts admit meta tensors on original and prepared model trees."""

import copy
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor.parallel import ParallelStyle

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.tensor_parallel.parallelize import resolve_tensor_parallel_plan
from torchturbo.engine.config import EngineConfig
from torchturbo.engine.model_fqn import build_model_fqn_map
from torchturbo.models.modules.moe.moe_layer import MoELayer
from torchturbo.parallel import TensorParallelValidator
from torchturbo.runtime.cpu import CpuRuntime


def _model_and_plan(family):
    if family == "qwen3":
        from tests.models.qwen3_moe.test_parallel_plan import _config
        from torchturbo.models.qwen3_moe.model import TurboQwen3MoeForCausalLM
        from torchturbo.models.qwen3_moe.parallel_plan import get_parallel_plan
        config = _config(num_hidden_layers=2)
        config._attn_implementation = "eager"
        return TurboQwen3MoeForCausalLM(config), get_parallel_plan()
    if family.startswith("qwen3.5"):
        from tests.models.qwen3_5_moe.tp_helpers import text_model
        model = text_model()
        if family.endswith("text"):
            from torchturbo.models.qwen3_5_moe_text.parallel_plan import get_parallel_plan
        else:
            from torchturbo.models.qwen3_5_moe.parallel_plan import get_parallel_plan
            language = model.model
            model.model = nn.Module()
            model.model.language_model = language
        return model, get_parallel_plan()
    from tests.models.bailing_hybrid.tp_helpers import make_bailing_model
    if family == "hybrid":
        from torchturbo.models.bailing_hybrid.parallel_plan import get_parallel_plan
    else:
        from torchturbo.models.bailing_moe_v3_vl.parallel_plan import get_parallel_plan
    return make_bailing_model(family), get_parallel_plan()


def _metadata_snapshot(model):
    return (
        {name: id(module) for name, module in model.named_modules()},
        {name: id(parameter) for name, parameter in model.named_parameters()},
        {name: {key: value for key, value in vars(module).items()
                if type(value) in (str, int, bool, float, type(None))}
         for name, module in model.named_modules()},
        {name: (dict(module._forward_hooks), dict(module._forward_pre_hooks))
         for name, module in model.named_modules()},
        copy.deepcopy(model.config.to_dict()),
    )


@pytest.mark.parametrize("family", ["qwen3", "qwen3.5-text", "qwen3.5-vl", "hybrid", "vl"])
def test_validator_is_read_only_on_original_and_prepared_meta_trees(tmp_path, family):
    from tests.models.bailing_hybrid.tp_helpers import cpu_bailing_leaves
    leaves = cpu_bailing_leaves() if family in ("hybrid", "vl") else nullcontext()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}",
                            rank=0, world_size=1)
    try:
        with leaves, torch.device("meta"):
            model, plan = _model_and_plan(family)
            validator = plan.tensor_parallel_plan.validator
            assert isinstance(validator, TensorParallelValidator)
            assert all(isinstance(style, ParallelStyle)
                       for style in plan.tensor_parallel_plan.plan.values())
            admission_plan = copy.copy(plan)
            admission_plan.tp_size = admission_plan.etp_size = 2
            admission_plan.sp_size = 2
            before = _metadata_snapshot(model)
            validator.validate(model, admission_plan)
            assert _metadata_snapshot(model) == before
            backend = FSDP2Backend(EngineConfig(), CpuRuntime())
            backend._init_mesh(plan)
            model = backend._replace_modules(model, plan)
            before = _metadata_snapshot(model)
            validator.validate(model, admission_plan)
            assert _metadata_snapshot(model) == before
            assert all(parameter.is_meta for parameter in model.parameters())
            assert any(isinstance(module, MoELayer) for module in model.modules())
            resolved = resolve_tensor_parallel_plan(admission_plan.tensor_parallel_plan.plan, build_model_fqn_map(model))
            assert all(not isinstance(style, TensorParallelValidator) for style in resolved.values())
    finally:
        dist.destroy_process_group()
