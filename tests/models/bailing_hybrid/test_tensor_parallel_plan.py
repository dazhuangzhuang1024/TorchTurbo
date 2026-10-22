"""Bailing actual inventory and cache admission before model mutation."""

import copy

import pytest
import torch
from torch import nn

from .tp_helpers import cpu_bailing_leaves, make_bailing_model


def family_plan(family):
    if family == "hybrid":
        from torchturbo.models.bailing_hybrid.parallel_plan import get_parallel_plan
    else:
        from torchturbo.models.bailing_moe_v3_vl.parallel_plan import get_parallel_plan
    plan = get_parallel_plan()
    assert plan.tensor_parallel_plan.plan, "Both Bailing families need native TP inventory"
    return plan


def validate_model(family, model, *, tp_size=2, sp_size=1):
    plan = family_plan(family)
    plan.tp_size = plan.etp_size = tp_size
    plan.sp_size = sp_size
    plan.tensor_parallel_plan.validator.validate(model, plan)


@pytest.mark.parametrize("family", ["hybrid", "vl"])
@pytest.mark.parametrize("schedule", [1, 2])
@pytest.mark.parametrize("shared", [None, 1])
def test_actual_inventory_preserves_default_cache_and_global_config(family, schedule, shared):
    with cpu_bailing_leaves(), torch.device("meta"):
        model = make_bailing_model(family, layer_group_size=schedule, num_shared_experts=shared)
    before = copy.deepcopy(model.model.config.to_dict())
    validate_model(family, model)
    assert model.model.config.to_dict() == before
    assert model.model.config.use_cache is True
    assert model.model.layers[0].mlp.__class__.__name__ == "BailingMoeV3MLP"
    assert model.model.layers[1].mlp.__class__.__name__ == "BailingMoeV3SparseMoeBlock"
    assert model.model.layers[0].attention_layer_type == ("linear_attention" if schedule == 2 else "attention")
    assert model.model.layers[1].attention_layer_type == "attention"


def test_vl_all_kda_inventory_is_admitted():
    with cpu_bailing_leaves(), torch.device("meta"):
        model = make_bailing_model("vl", layer_group_size=[1, 1])
    validate_model("vl", model)
    assert model.model.config.full_attention_layer_ids == []


@pytest.mark.parametrize("family", ["hybrid", "vl"])
@pytest.mark.parametrize("broken,message", [
    ("heads", "heads"), ("mla", "q_proj"), ("kda_q", "q_proj"),
    ("conv", "conv"), ("conv_bias", "bias"), ("norm", "o_norm"),
    ("decay", "A_log"), ("dt_bias", "dt_bias"),
    ("mla_kv_a", "kv_a_proj"), ("mla_kv_b", "kv_b_proj"),
    ("mla_norm", "kv_a_layernorm"), ("kda_k", "k_proj"),
    ("kda_v", "v_proj"), ("kda_f", "f_b_proj"), ("kda_g", "g_b_proj"),
    ("kda_output", "o_proj"), ("shared", "shared"), ("routed", "expert"),
    ("gate", "gate"), ("dropout", "dropout"), ("mtp", "MTP"),
    ("dense", "dense"), ("no_routed", "routed"),
])
def test_invalid_inventory_fails_before_mesh_or_replacement(monkeypatch, family, broken, message):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import EngineConfig
    options = dict(num_experts=None) if broken == "no_routed" else {}
    if broken == "heads":
        options["num_attention_heads"] = 3
    with cpu_bailing_leaves(), torch.device("meta"):
        model = make_bailing_model(family, **options)
        linear, mla = model.model.layers
        if broken == "heads":
            linear.attention.num_heads = 3
        elif broken == "mla":
            mla.attention.q_proj = nn.Linear(16, 47, bias=False)
        elif broken == "kda_q":
            linear.attention.q_proj = nn.Linear(16, 15, bias=False)
        elif broken == "conv":
            linear.attention.q_conv1d.weight = nn.Parameter(torch.empty(15, 1, 4))
        elif broken == "conv_bias":
            linear.attention.q_conv1d.bias = nn.Parameter(torch.empty(16))
        elif broken == "norm":
            linear.attention.o_norm.weight = nn.Parameter(torch.empty(3))
        elif broken == "decay":
            linear.attention.A_log = nn.Parameter(torch.empty(3))
        elif broken == "dt_bias":
            linear.attention.dt_bias = nn.Parameter(torch.empty(15))
        elif broken == "mla_kv_a":
            mla.attention.kv_a_proj_with_mqa = nn.Linear(16, 15, bias=False)
        elif broken == "mla_kv_b":
            mla.attention.kv_b_proj = nn.Linear(8, 31, bias=False)
        elif broken == "mla_norm":
            mla.attention.kv_a_layernorm.weight = nn.Parameter(torch.empty(7))
        elif broken == "kda_k":
            linear.attention.k_proj = nn.Linear(16, 15, bias=False)
        elif broken == "kda_v":
            linear.attention.v_proj = nn.Linear(16, 15, bias=False)
        elif broken == "kda_f":
            linear.attention.f_b_proj = nn.Linear(4, 15, bias=False)
        elif broken == "kda_g":
            linear.attention.g_b_proj = nn.Linear(4, 15, bias=False)
        elif broken == "kda_output":
            linear.attention.o_proj = nn.Linear(15, 16, bias=False)
        elif broken == "shared":
            mla.mlp.shared_experts.up_proj = nn.Linear(16, 11, bias=False)
        elif broken == "routed":
            mla.mlp.experts[0].up_proj = nn.Linear(16, 11, bias=False)
        elif broken == "gate":
            mla.attention.gated_attention_proj_granularity_type = "unsupported"
        elif broken == "dropout":
            mla.attention.attention_dropout = 0.1
        elif broken == "mtp":
            if family == "hybrid":
                from torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3 import BailingMoeV3MTPLayer
            else:
                from torchturbo.models.bailing_moe_v3_vl.modeling_bailing_moe_v3_vl import BailingMoeV3MTPLayer
            model.model.layers.append(BailingMoeV3MTPLayer(model.model.config, 1))
        elif broken == "dense":
            linear.mlp.up_proj = nn.Linear(16, 23, bias=False)
    plan = model.get_default_parallel_plan()
    plan.tp_size = plan.etp_size = 2
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.config = EngineConfig()
    backend.fsdp_config = backend.config.backend_config
    backend._parallel_context = None
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(backend, "_init_mesh", lambda *args: calls.append("mesh"))
    monkeypatch.setattr(backend, "_replace_modules", lambda *args: calls.append("replace"))
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    config = copy.deepcopy(model.model.config.to_dict())
    with pytest.raises(ValueError, match=message):
        backend.prepare_model(model, plan)
    assert calls == []
    assert identities == {name: id(parameter) for name, parameter in model.named_parameters()}
    assert model.model.config.to_dict() == config


@pytest.mark.parametrize("shared", [None, 1])
@pytest.mark.parametrize("family", ["hybrid", "vl"])
def test_prepared_sparse_inventory_and_vl_context_argument(tmp_path, family, shared):
    import torch.distributed as dist
    from torchturbo.engine.config import EngineConfig
    from torchturbo.models.modules.moe.experts import GroupedMLP
    from torchturbo.models.modules.moe.module import ModuleSpec
    from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
    dist.init_process_group("gloo", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1)
    try:
        with cpu_bailing_leaves(), torch.device("meta"):
            model = make_bailing_model(family, num_shared_experts=shared)
        plan = model.get_default_parallel_plan()
        replacement = plan.plan["model.layers.*.mlp"].replace_module
        kwargs = dict(hf_config=model.config, sp_info=dict(sp_group=None),
                      ep_info=dict(ep_group=dist.group.WORLD, ep_size=1),
                      engine_config=EngineConfig(),
                      moe_submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)))
        dense = model.model.layers[0].mlp
        assert replacement(dense, **kwargs) is dense
        assert replacement(dense, parallel_context={}, **kwargs) is dense
        original = model.model.layers[1].mlp
        saved_shared = original.shared_experts if shared is not None else None
        assert isinstance(replacement(original, **kwargs), MoELayer)
        model.model.layers[1].mlp = replacement(original, parallel_context={}, **kwargs)
        prepared = model.model.layers[1].mlp
        assert isinstance(prepared, MoELayer)
        assert prepared.shared_experts is saved_shared
        assert prepared.layer_number == 1
        validate_model(family, model)
    finally:
        dist.destroy_process_group()


def test_hybrid_registration_and_tp1_plan_survive_unavailable_vl():
    import subprocess
    import sys
    source = '''
import importlib.abc
import sys
class BlockVL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("torchturbo.models.bailing_moe_v3_vl"):
            raise ImportError("Deliberately unavailable unrelated VL family")
sys.meta_path.insert(0, BlockVL())
from torchturbo.models.registry import get_model
model = get_model("bailing_hybrid")
assert model is not None
plan = model.get_default_parallel_plan()
assert plan.tp_size == plan.etp_size == 1
assert "" in plan.tensor_parallel_plan.plan
assert not any(name.startswith("torchturbo.models.bailing_moe_v3_vl") for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("family", ["hybrid", "vl"])
def test_shared_cache_guard_uses_actual_root_and_backbone_signatures(family):
    from torchturbo.engine.tensor_parallel.parallelize import register_tensor_parallel_cache_guard
    with cpu_bailing_leaves(), torch.device("meta"):
        model = make_bailing_model(family, layer_group_size=1)
    register_tensor_parallel_cache_guard(model)
    register_tensor_parallel_cache_guard(model.model)
    calls = []
    def computation(*args, **kwargs):
        calls.append("computation")
        raise RuntimeError("Reached computation")
    handles = [model.model.word_embeddings.register_forward_pre_hook(computation)]
    if family == "vl":
        handles.append(model.model.visual.register_forward_pre_hook(computation))
    try:
        inputs = torch.ones(1, 2, dtype=torch.long)
        for module in (model, model.model):
            for kwargs in ({"use_cache": True}, {"past_key_values": object()}):
                with pytest.raises(ValueError, match="cache"):
                    module(input_ids=inputs, **kwargs)
            with pytest.raises(ValueError, match="cache"):
                module(inputs, None, None, object())
            from inspect import signature
            parameters = list(signature(module.forward).parameters)
            if "use_cache" in parameters:
                positional = [inputs] + [None] * (parameters.index("use_cache") - 1) + [True]
                with pytest.raises(ValueError, match="cache"):
                    module(*positional)
        assert not calls
        if family == "vl":
            with pytest.raises(ValueError, match="cache"):
                model(input_ids=inputs, use_cache=True, pixel_values=torch.ones(4, 24),
                      image_grid_thw=torch.tensor([[1, 2, 2]]))
        for kwargs in ({"use_cache": False}, {"use_cache": None}, {}):
            with pytest.raises(RuntimeError, match="Reached computation"):
                model(input_ids=inputs, **kwargs)
    finally:
        for handle in handles:
            handle.remove()


def test_vl_all_kda_cacheless_entry_reaches_actual_decoder():
    with cpu_bailing_leaves():
        model = make_bailing_model("vl", layer_group_size=[1, 1])
        calls = []
        def stop_before_kda(*args, **kwargs):
            calls.append("decoder")
            raise RuntimeError("Reached actual KDA decoder")
        handle = model.model.layers[0].register_forward_pre_hook(stop_before_kda)
        try:
            with pytest.raises(RuntimeError, match="Reached actual KDA decoder"):
                model(input_ids=torch.ones(1, 2, dtype=torch.long), use_cache=False)
            assert calls == ["decoder"]
        finally:
            handle.remove()


@pytest.mark.parametrize("family", ["hybrid", "vl"])
def test_configured_nonzero_text_dropout_fails_admission(family):
    schedule = 4 if family == "hybrid" else [1, 1]
    with cpu_bailing_leaves(), torch.device("meta"):
        model = make_bailing_model(family, layer_group_size=schedule, attention_dropout=0.1)
    with pytest.raises(ValueError, match="attention_dropout"):
        validate_model(family, model)


@pytest.mark.parametrize("family", ["hybrid", "vl"])
@pytest.mark.parametrize("no_kda_lora", [False, True])
@pytest.mark.parametrize("query_rank,gate", [(None, None), (8, "head_wise"), (8, "element_wise")])
def test_constructor_selected_projection_variants(family, no_kda_lora, query_rank, gate):
    with cpu_bailing_leaves(), torch.device("meta"):
        model = make_bailing_model(family, no_kda_lora=no_kda_lora,
                                   q_lora_rank=query_rank,
                                   gated_attention_proj_granularity_type=gate,
                                   use_qkv_bias=True)
    validate_model(family, model, sp_size=2)
    assert model.model.layers[0].attention.no_kda_lora is no_kda_lora


@pytest.mark.parametrize("family", ["hybrid", "vl"])
@pytest.mark.parametrize("field", ["intermediate_size", "moe_intermediate_size", "moe_shared_expert_intermediate_size"])
def test_configured_feature_width_must_be_divisible(family, field):
    with cpu_bailing_leaves(), torch.device("meta"):
        model = make_bailing_model(family, **{field: 11})
    with pytest.raises(ValueError, match="features must be divisible"):
        validate_model(family, model)


@pytest.mark.parametrize("family", ["hybrid", "vl"])
def test_tp1_owned_config_retains_cache_behavior_without_ac(family):
    with cpu_bailing_leaves(), torch.device("meta"):
        first = make_bailing_model(family, layer_group_size=1)
        second = make_bailing_model(family, layer_group_size=1, use_cache=False)
    assert first.model.config is not second.model.config
    assert not first.model.gradient_checkpointing
    assert not second.model.gradient_checkpointing
    assert first.get_default_parallel_plan().tp_size == 1
    validate_model(family, first, tp_size=1)
    validate_model(family, second, tp_size=1)
    calls = []
    def stop(*args, **kwargs):
        calls.append("embedding")
        raise RuntimeError("Reached TP1 embedding")
    handle = first.model.word_embeddings.register_forward_pre_hook(stop)
    try:
        with pytest.raises(RuntimeError, match="TP1 embedding"):
            first(input_ids=torch.ones(1, 2, dtype=torch.long), use_cache=True)
        assert calls == ["embedding"]
        assert first.model.config.use_cache is True
        assert second.model.config.use_cache is False
    finally:
        handle.remove()


@pytest.mark.parametrize("family", ["hybrid", "vl"])
@pytest.mark.parametrize("options", [
    dict(rope_interleave=False), dict(v_head_dim=16),
    dict(hidden_act="gelu", scoring_func="softmax", router_dtype="bf16"),
    dict(num_experts_per_tok=1, routed_scaling_factor=0),
    dict(expert_swiglu_limit_list=[-1.0, -1.0]),
    dict(head_dim=300, safe_gate=True, lower_bound=0),
])
def test_source_and_leaf_owned_limits_do_not_become_tp_admission(family, options):
    with cpu_bailing_leaves(), torch.device("meta"):
        model = make_bailing_model(family, **options)
    validate_model(family, model)
