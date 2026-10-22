"""Qwen authors only real TP targets and validates complete head partitions."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention, Qwen3MoeMLP

from torchturbo.models.qwen3_moe.parallel_plan import get_parallel_plan, QWen3MoeLayerWrapper


def _config(**options):
    values = dict(hidden_size=16, num_attention_heads=8, num_key_value_heads=4,
                  head_dim=6, attention_dropout=0.0, num_experts=4,
                  moe_intermediate_size=8, dtype=torch.float32)
    values.update(options)
    return Qwen3MoeConfig(**values)


@pytest.mark.parametrize('options,tp_size,broken,message', [
    ({}, 2, None, None),
    ({'num_attention_heads': 6}, 4, None, 'q_proj'),
    ({'num_key_value_heads': 2}, 4, None, 'k_proj'),
    ({}, 2, 'v_proj', 'v_proj'),
    ({'attention_dropout': .1}, 2, None, 'attention_dropout'),
])
def test_actual_head_width_validation(options, tp_size, broken, message):
    plan = get_parallel_plan()
    assert 'model.layers.*.self_attn' not in plan.tensor_parallel_plan.plan
    config = _config(**options)
    original_config = copy.deepcopy(config.to_dict())
    with torch.device('meta'):
        attention = Qwen3MoeAttention(config, layer_idx=0)
        if broken:
            attention.v_proj = nn.Linear(16, 10, bias=False)
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = attention
    with torch.device('meta'):
        layer.mlp = Qwen3MoeSparseMoeBlock(config)
    model.model.layers = nn.ModuleList([layer])
    plan.tp_size = plan.etp_size = tp_size
    validator = plan.tensor_parallel_plan.validator
    if message:
        with pytest.raises(ValueError, match=message):
            validator.validate(model, plan)
    else:
        validator.validate(model, plan)
    assert config.to_dict() == original_config
    assert 'num_heads' not in vars(attention)
    assert 'num_key_value_heads' not in vars(attention)
    assert not any('shared_expert' in key for key in plan.tensor_parallel_plan.plan)


@pytest.mark.parametrize('num_key_value_heads,sp_size', [(2, 2), (4, 2), (6, 1)])
def test_actual_tp_local_kv_sp_admission_preserves_metadata(
    monkeypatch, num_key_value_heads, sp_size
):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import EngineConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    config = _config(num_hidden_layers=1, num_attention_heads=12,
                     num_key_value_heads=num_key_value_heads, head_dim=4)
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = Qwen3MoeForCausalLM(config)
    before = copy.deepcopy(config.to_dict())
    original_mlp = model.model.layers[0].mlp
    original_parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.config = EngineConfig()
    backend.fsdp_config = backend.config.backend_config
    backend._parallel_context = None
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2 * sp_size)
    plan = get_parallel_plan()
    plan.tp_size = plan.etp_size = 2
    plan.sp_size = sp_size
    backend._validate_tensor_parallel_features(model, plan, None)
    assert model.model.layers[0].mlp is original_mlp
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == original_parameters
    assert config.to_dict() == before
    assert 'num_heads' not in vars(model.model.layers[0].self_attn)
    assert 'num_key_value_heads' not in vars(model.model.layers[0].self_attn)


def test_actual_tp_local_kv_sp_rejected_before_mesh_or_replacement(monkeypatch):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import EngineConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    config = _config(num_hidden_layers=1, num_attention_heads=12,
                     num_key_value_heads=6, head_dim=4)
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = Qwen3MoeForCausalLM(config)
    before = copy.deepcopy(config.to_dict())
    original_mlp = model.model.layers[0].mlp
    attention = model.model.layers[0].self_attn
    original_metadata = vars(attention).copy()
    original_parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.config = EngineConfig()
    backend.fsdp_config = backend.config.backend_config
    backend._parallel_context = None
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 4)

    def fail_if_mutation_reached(*args):
        pytest.fail('Incompatible TP-local KV/SP reached mesh creation or replacement.')

    monkeypatch.setattr(backend, '_init_mesh', fail_if_mutation_reached)
    monkeypatch.setattr(backend, '_replace_modules', fail_if_mutation_reached)
    plan = get_parallel_plan()
    plan.tp_size = plan.etp_size = plan.sp_size = 2
    with pytest.raises(ValueError, match='KV heads.*SP'):
        backend.prepare_model(model, plan)
    assert backend._parallel_context is None
    assert model.model.layers[0].mlp is original_mlp
    assert vars(attention) == original_metadata
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == original_parameters
    assert config.to_dict() == before
    plan.tp_size = plan.etp_size = 1
    backend._validate_tensor_parallel_features(model, plan, None)


def test_actual_dense_mlp_rejected_before_new_path_replacement(monkeypatch):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import EngineConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    config = _config(mlp_only_layers=[1], num_hidden_layers=2)
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = Qwen3MoeForCausalLM(config)
    original = model.model.layers[0].mlp
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.config = EngineConfig()
    backend.fsdp_config = backend.config.backend_config
    backend._parallel_context = None
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    plan = get_parallel_plan()
    plan.tp_size = plan.etp_size = 2
    with pytest.raises(ValueError, match='all-routed.*layer 1'):
        backend.prepare_model(model, plan)
    assert model.model.layers[0].mlp is original
    plan.tp_size = plan.etp_size = 1
    backend._validate_tensor_parallel_features(model, plan, None)


def test_wrapper_preserves_attached_shared_experts(tmp_path):
    import torch.distributed as dist
    from torchturbo.models.modules.moe.experts import GroupedMLP
    from torchturbo.models.modules.moe.moe_layer import MoESubmodules

    dist.init_process_group('gloo', init_method=f'file://{tmp_path / "store"}',
                            rank=0, world_size=1)
    try:
        original = nn.Module()
        original.shared_experts = nn.Linear(16, 16, bias=False)
        wrapper = QWen3MoeLayerWrapper(
            original_module=original,
            hf_config=_config(num_experts_per_tok=2),
            sp_info={}, ep_info={'ep_group': dist.group.WORLD, 'ep_size': 1},
            moe_submodules=MoESubmodules(experts=GroupedMLP),
        )
        assert wrapper.shared_experts is original.shared_experts
        assert wrapper.shared_experts.weight is original.shared_experts.weight
    finally:
        dist.destroy_process_group()


def _prepared_qwen_worker(rank, path):
    from datetime import timedelta
    import torch.distributed as dist
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import EngineConfig, MixedPrecisionPolicy
    from torchturbo.engine.model_fqn import build_model_fqn_map
    from torchturbo.models.qwen3_moe.model import TurboQwen3MoeForCausalLM
    from torchturbo.models.modules.moe.moe_layer import MoELayer
    from torchturbo.engine.tensor_parallel.parallelize import resolve_tensor_parallel_plan
    from torchturbo.runtime.cpu import CpuRuntime
    dist.init_process_group('gloo', init_method=f'file://{path}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        config = _config(num_hidden_layers=2, decoder_sparse_step=1, mlp_only_layers=[7])
        config._attn_implementation = 'eager'
        before = copy.deepcopy(config.to_dict())
        engine_config = EngineConfig(mixed_precision=MixedPrecisionPolicy.FP32)
        engine_config.moe.aux_loss_coeff = .03
        engine_config.moe.z_loss_coeff = .004
        backend = FSDP2Backend(engine_config, CpuRuntime())
        plan = get_parallel_plan()
        plan.tp_size = plan.etp_size = 2
        with torch.device('meta'):
            model = TurboQwen3MoeForCausalLM(config)
            backend.prepare_model(model, plan)
        resolved = resolve_tensor_parallel_plan(plan.tensor_parallel_plan.plan, build_model_fqn_map(model))
        again = resolve_tensor_parallel_plan(plan.tensor_parallel_plan.plan, build_model_fqn_map(model))
        assert len(resolved) == 14
        assert 'model.layers' not in resolved
        for index, layer in enumerate(model.model.layers):
            assert isinstance(layer.mlp, MoELayer)
            parent = f'model.layers.{index}.self_attn'
            assert parent not in resolved
            assert f'{parent}.q_proj' in resolved
            assert 'num_heads' not in vars(layer.self_attn)
            assert 'num_key_value_heads' not in vars(layer.self_attn)
            assert layer.mlp.config.moe_aux_loss_coeff == .03
            assert layer.mlp.config.moe_z_loss_coeff == .004
            assert layer.mlp.router.routing_group is backend.get_etp_info()['etp_group']
            assert layer.mlp.router.sp_group is backend.get_sp_info()['sp_group']
            assert backend._parameter_ownership.is_expert(layer.mlp.experts.weight1)
        for name, style in resolved.items():
            assert style is not again[name]
        assert config.to_dict() == before
    finally:
        dist.destroy_process_group()


def test_real_qwen_prepared_tree_and_runtime_injection(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_prepared_qwen_worker, args=(str(tmp_path/'store'),), nprocs=2, join=True)
