"""Admission of actual hybrid decoder shapes before TP/replacement mutation."""

import copy
from datetime import timedelta

import pytest
import torch
from torch import nn

from torchturbo.engine.model_fqn import build_model_fqn_map
from torchturbo.models.qwen3_5_moe.parallel_plan import get_parallel_plan as vlm_plan
from torchturbo.models.qwen3_5_moe_text.parallel_plan import get_parallel_plan as text_plan
from torchturbo.engine.tensor_parallel.parallelize import resolve_tensor_parallel_plan

from .tp_helpers import model_validator, text_model


@pytest.mark.parametrize("types", [
    ["full_attention", "full_attention"],
    ["linear_attention", "linear_attention"],
    ["linear_attention", "full_attention"],
])
@pytest.mark.parametrize("wrapper", ["text", "vlm"])
def test_actual_optional_mixers_resolve_and_validate_without_config_mutation(types, wrapper):
    with torch.device("meta"):
        model = text_model(layer_types=types)
    if wrapper == "vlm":
        language_model = model.model
        model.model = nn.Module()
        model.model.language_model = language_model
        plan = vlm_plan()
    else:
        plan = text_plan()
    plan.tp_size = plan.etp_size = 2
    before = copy.deepcopy(model.config.to_dict())
    first = resolve_tensor_parallel_plan(plan.tensor_parallel_plan.plan, build_model_fqn_map(model))
    second = resolve_tensor_parallel_plan(plan.tensor_parallel_plan.plan, build_model_fqn_map(model))
    assert first, "Both registered models need a nonempty language TP plan"
    plan.tensor_parallel_plan.validator.validate(model, plan)
    for fqn, style in first.items():
        assert style is not second[fqn]
    assert model.config.to_dict() == before


@pytest.mark.parametrize("broken,message", [
    ("q_gate", "q_proj"), ("k", "k_proj"), ("v", "v_proj"),
    ("out", "o_proj"), ("qkv", "in_proj_qkv"), ("conv", "conv1d"),
    ("time", "A_log"), ("z", "in_proj_z"), ("shared", "shared_expert"),
    ("ratio", "value.*key|num_v_heads"), ("mtp", "MTP"),
])
def test_malformed_actual_shapes_rejected_before_metadata_or_replacement(broken, message):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import EngineConfig

    with torch.device("meta"):
        model = text_model()
        linear, full = model.model.layers
        if broken == "q_gate":
            full.self_attn.q_proj = nn.Linear(16, 48, bias=False)
        elif broken in ("k", "v"):
            setattr(full.self_attn, f"{broken}_proj", nn.Linear(16, 8, bias=False))
        elif broken == "out":
            full.self_attn.o_proj = nn.Linear(24, 16, bias=False)
        elif broken == "qkv":
            linear.linear_attn.in_proj_qkv = nn.Linear(16, 62, bias=False)
        elif broken == "conv":
            linear.linear_attn.conv1d = nn.Conv1d(62, 62, 3, groups=62, bias=False)
        elif broken == "time":
            linear.linear_attn.A_log = nn.Parameter(torch.zeros(6))
        elif broken == "z":
            linear.linear_attn.in_proj_z = nn.Linear(16, 28, bias=False)
        elif broken == "shared":
            full.mlp.shared_expert.up_proj = nn.Linear(16, 11, bias=False)
        elif broken == "ratio":
            linear.linear_attn.num_v_heads = 6
        else:
            model.mtp = nn.Linear(16, 16)
    plan = text_plan()
    plan.tp_size = plan.etp_size = 2
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.config = EngineConfig()
    backend.fsdp_config = backend.config.backend_config
    backend._parallel_context = None
    original = linear.mlp
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    try:
        with pytest.raises(ValueError, match=message):
            backend._validate_tensor_parallel_features(model, plan, None)
    finally:
        monkeypatch.undo()
    assert linear.mlp is original
    assert linear.linear_attn.num_k_heads == 4


@pytest.mark.parametrize("key_heads,value_heads,sp_size,accepted", [
    (8, 16, 2, True), (4, 8, 2, True), (2, 4, 2, True),
    (6, 12, 2, False), (2, 6, 2, False), (4, 8, 3, False),
])
def test_tp_local_sp_grouping_admission(key_heads, value_heads, sp_size, accepted):
    with torch.device("meta"):
        model = text_model(layer_types=["linear_attention", "linear_attention"],
                           linear_num_key_heads=key_heads,
                           linear_num_value_heads=value_heads)
    validator, plan = model_validator(text_plan(), sp_size=sp_size)
    if accepted:
        validator.validate(model, plan)
    else:
        with pytest.raises(ValueError, match="SP|sequence parallel"):
            validator.validate(model, plan)
    assert model.model.layers[0].linear_attn.num_k_heads == key_heads


def test_invalid_tp_local_sp_rejected_before_replacement(monkeypatch):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import EngineConfig

    with torch.device("meta"):
        model = text_model(layer_types=["linear_attention", "linear_attention"],
                           linear_num_key_heads=6, linear_num_value_heads=12)
    original = model.model.layers[0].mlp
    plan = text_plan()
    plan.tp_size = plan.etp_size = plan.sp_size = 2
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.config = EngineConfig()
    backend.fsdp_config = backend.config.backend_config
    backend._parallel_context = None
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    with pytest.raises(ValueError, match="SP"):
        backend.prepare_model(model, plan)
    assert model.model.layers[0].mlp is original
    assert model.model.layers[0].linear_attn.num_k_heads == 6


def test_full_attention_tp_local_sp_rejection_preserves_metadata():
    with torch.device("meta"):
        model = text_model(num_attention_heads=12, num_key_value_heads=6,
                           layer_types=["full_attention", "full_attention"])
    with pytest.raises(ValueError, match="KV heads.*SP"):
        validator, plan = model_validator(text_plan(), sp_size=2)
        validator.validate(model, plan)
    assert model.model.layers[0].self_attn.config.num_key_value_heads == 6


def test_normal_sp_preparation_is_instance_scoped_and_tp_s1_registers_attention():
    from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet

    class_forward = Qwen3_5MoeGatedDeltaNet.forward
    legacy = text_model()
    legacy.on_parallelized({"sp": {"sp_size": 2, "sp_group": None, "sp_rank": 0},
                            "tp": {"tp_size": 1}})
    assert legacy.model.layers[0].linear_attn.forward.__func__ is not class_forward
    assert Qwen3_5MoeGatedDeltaNet.forward is class_forward
    tp_s1 = text_model()
    tp_s1.on_parallelized({"sp": {"sp_size": 1, "sp_group": None, "sp_rank": 0},
                           "tp": {"tp_size": 2}})
    assert tp_s1.config._attn_implementation == "turbo_ulysses"
    assert tp_s1.model.layers[0].linear_attn.forward.__func__ is class_forward
    tp_sp = text_model()
    tp_sp.on_parallelized({"sp": {"sp_size": 2, "sp_group": None, "sp_rank": 1},
                           "tp": {"tp_size": 2}})
    assert tp_sp.config._attn_implementation == "turbo_ulysses"
    reference = text_model()
    assert reference.config._attn_implementation == "eager"
    assert reference.model.layers[0].linear_attn.forward.__func__ is class_forward


def _prepared_worker(rank, rendezvous):
    import torch.distributed as dist
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import EngineConfig, MixedPrecisionPolicy
    from torchturbo.models.modules.moe.moe_layer import MoELayer
    from torchturbo.runtime.cpu import CpuRuntime

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        engine_config = EngineConfig(mixed_precision=MixedPrecisionPolicy.FP32)
        backend = FSDP2Backend(engine_config, CpuRuntime())
        plan = text_plan()
        plan.tp_size = plan.etp_size = 2
        with torch.device("meta"):
            model = text_model()
            before = copy.deepcopy(model.config.to_dict())
            shared = [layer.mlp.shared_expert for layer in model.model.layers]
            shared_gates = [layer.mlp.shared_expert_gate for layer in model.model.layers]
            backend.prepare_model(model, plan)
        resolved = resolve_tensor_parallel_plan(plan.tensor_parallel_plan.plan, build_model_fqn_map(model))
        assert list(resolved)[0] == ""
        for index, layer in enumerate(model.model.layers):
            assert isinstance(layer.mlp, MoELayer)
            assert layer.mlp.shared_expert is shared[index]
            assert layer.mlp.shared_expert_gate is shared_gates[index]
            assert backend._parameter_ownership.is_expert(layer.mlp.experts.weight1)
            assert not backend._parameter_ownership.is_expert(layer.mlp.shared_expert.gate_proj.weight)
            assert not backend._parameter_ownership.is_expert(layer.mlp.shared_expert_gate.weight)
            assert layer.mlp.router.routing_group is backend.get_etp_info()["etp_group"]
        assert model.config.to_dict() == before
    finally:
        dist.destroy_process_group()


def test_real_registered_preparation_preserves_shared_module_ownership(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_prepared_worker, args=(str(tmp_path / "prepared-store"),), nprocs=2, join=True)
