"""Shared grouped experts use paired feature shards and native local kernels."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.tensor import DTensor, Shard

from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext


def _experts(num_experts=3, **options):
    config = dict(hidden_size=3, moe_ffn_hidden_size=4, gated_linear_unit=True,
                  activation_func=nn.functional.silu, params_dtype=torch.float32)
    config.update(options)
    model = GroupedMLP(num_experts, False, MoEConfig(**config))
    with torch.no_grad():
        model.weight1.copy_(torch.arange(model.weight1.numel()).reshape_as(model.weight1) / 37 - 1)
        model.weight2.copy_(torch.arange(model.weight2.numel()).reshape_as(model.weight2) / 29 - .5)
    return model


def _reference(x, w1, w2, counts):
    outputs = []
    offset = 0
    for expert, count in enumerate(counts):
        gate, up = (x[offset:offset + count] @ w1[expert]).chunk(2, dim=-1)
        outputs.append((nn.functional.silu(gate) * up) @ w2[expert])
        offset += count
    return torch.cat(outputs)


def test_legacy_experts_keep_packed_weights():
    model = _experts()
    assert model.weight1.shape == (9, 8)
    assert model.weight2.shape == (12, 3)


def _expert_worker(rank, path, mode):
    dist.init_process_group('gloo', init_method=f'file://{path}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        from torchturbo.engine.tensor_parallel.expert import GroupedMLPParallel
        context = ParallelContext('cpu', sp_size=1, ep_size=1, tp_size=2, etp_size=2)
        model = _experts()
        w1 = model.weight1.detach().clone().view(3, 3, 8).requires_grad_()
        w2 = model.weight2.detach().clone().view(3, 4, 3).requires_grad_()
        identity = id(model)
        GroupedMLPParallel()._apply(model, context.etp_mesh)
        assert id(model) == identity
        assert model.weight1.shape == (9, 2, 4)
        assert model.weight2.shape == (3, 4, 3)
        assert model.weight1.placements == (Shard(2),)
        assert model.weight2.placements == (Shard(1),)
        torch.testing.assert_close(model.weight1.to_local(), w1.view(9, 2, 4)[:, :, rank*2:(rank+1)*2])
        torch.testing.assert_close(model.weight2.to_local(), w2[:, rank*2:(rank+1)*2, :])
        if mode == 'slices':
            return
        from torchturbo.ops import get_or_create_ops
        from torchturbo.runtime.cpu import CpuRuntime
        get_or_create_ops(device_runtime=CpuRuntime())
        from torchturbo.models.modules.moe import experts as expert_module
        real_grouped_matmul = expert_module.ops.grouped_matmul
        calls = []
        def check_local(x, weight, counts, **kwargs):
            assert not isinstance(x, DTensor)
            assert not isinstance(weight, DTensor)
            calls.append(tuple(weight.shape))
            return real_grouped_matmul(x, weight, counts, **kwargs)
        expert_module.ops.grouped_matmul = check_local
        for step, counts in enumerate(([2, 0, 3], [0, 0, 0], [1, 3, 0])):
            model.zero_grad(set_to_none=True)
            w1.grad = w2.grad = None
            x = (torch.arange(sum(counts)*3.).view(-1, 3)/17 + step*.2).requires_grad_()
            ref_x = x.detach().clone().requires_grad_()
            probs = torch.linspace(.2, .8, sum(counts), requires_grad=True)
            ref_probs = probs.detach().clone().requires_grad_()
            output, bias = model(x, torch.tensor(counts), probs)
            expected = _reference(ref_x, w1, w2, counts)
            assert bias is None
            torch.testing.assert_close(output, expected)
            (output * probs[:, None]).square().sum().backward()
            (expected * ref_probs[:, None]).square().sum().backward()
            torch.testing.assert_close(x.grad, ref_x.grad)
            torch.testing.assert_close(probs.grad, ref_probs.grad)
            torch.testing.assert_close(model.weight1.grad.full_tensor(), w1.grad.view(9, 2, 4))
            torch.testing.assert_close(model.weight2.grad.full_tensor(), w2.grad)
        assert len(calls) == 4
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['slices', 'numeric'])
def test_aligned_experts(tmp_path, mode):
    mp.spawn(_expert_worker, args=(str(tmp_path / 'store'), mode), nprocs=2, join=True)


class SharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(3, 4, bias=False)
        self.up = nn.Linear(3, 4, bias=False)
        self.down = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


class SharedMoEModel(nn.Module):
    def __init__(self, ep_group, **moe_options):
        from types import SimpleNamespace
        from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
        from torchturbo.models.modules.moe.module import ModuleSpec
        super().__init__()
        self.config = SimpleNamespace()
        config = dict(hidden_size=3, moe_ffn_hidden_size=4,
            num_moe_experts=1, gated_linear_unit=True, activation_func=nn.functional.silu,
            expert_model_parallel_size=dist.get_world_size(ep_group),
            moe_router_topk=1, moe_router_score_function='sigmoid',
            moe_token_dispatcher_type='alltoall')
        config.update(moe_options)
        self.moe = MoELayer(MoEConfig(**config), ep_group,
            submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)))
        self.moe.shared_expert = SharedExpert()
        self.moe.shared_expert_gate = nn.Linear(3, 1, bias=False)
        self.norm = nn.RMSNorm(3)
        self.head = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.linspace(-.3, .5, parameter.numel()).reshape_as(parameter))

    def get_input_embeddings(self):
        raise NotImplementedError

    def get_lm_head(self):
        return self.head

    def get_norm(self):
        return self.norm

    def forward(self, x):
        return self.head(self.norm(self.moe(x)[0]))


def _fsdp_worker(rank, path, world):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import EngineConfig, MixedPrecisionPolicy
    from torchturbo.ops import get_or_create_ops
    from torchturbo.parallel import (
        TensorParallelPlanSpec,
        FSDPParallelPlan,
        FSDPPlanSpec,
        FSDPPlacementSpec,
    )
    from torchturbo.engine.tensor_parallel.expert import GroupedMLPParallel
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    from torchturbo.runtime.cpu import CpuRuntime
    dist.init_process_group('gloo', init_method=f'file://{path}', rank=rank,
                            world_size=world, timeout=timedelta(seconds=120))
    try:
        get_or_create_ops(device_runtime=CpuRuntime())
        context = ParallelContext('cpu', sp_size=1, ep_size=1, tp_size=2, etp_size=2)
        model = SharedMoEModel(context.ep_mesh.get_group())
        reference = SharedMoEModel(context.ep_mesh.get_group())
        experts = model.moe.experts
        plan = FSDPParallelPlan(tp_size=2, etp_size=2, plan={
            'moe.experts': FSDPPlanSpec(placement=FSDPPlacementSpec(Shard(0), 'edp')),
        }, tensor_parallel_plan=TensorParallelPlanSpec(plan={
            'moe.experts': GroupedMLPParallel(),
            'moe.shared_expert.gate': ColwiseParallel(),
            'moe.shared_expert.up': ColwiseParallel(),
            'moe.shared_expert.down': RowwiseParallel(input_layouts=Shard(-1)),
        }))
        engine_config = EngineConfig(mixed_precision=MixedPrecisionPolicy.FP32)
        engine_config.moe.aux_loss_coeff = .02
        engine_config.moe.z_loss_coeff = .003
        backend = FSDP2Backend(engine_config, CpuRuntime())
        backend._parallel_context = context
        backend._world_mesh = context.dense_fsdp_mesh
        backend._mesh = context.expert_mesh
        import copy
        from torchturbo.engine.model_fqn import build_model_fqn_map
        for invalid in ('missing_style', 'missing_expert_unit', 'dense_edp'):
            invalid_plan = copy.deepcopy(plan)
            if invalid == 'missing_style':
                del invalid_plan.tensor_parallel_plan.plan['moe.experts']
            elif invalid == 'missing_expert_unit':
                del invalid_plan.plan['moe.experts']
            else:
                invalid_plan.plan['norm'] = FSDPPlanSpec(placement=FSDPPlacementSpec(Shard(0), 'edp'))
            backend._resolved_parallel_plan = backend._compile_parallel_plan(
                invalid_plan, build_model_fqn_map(model)
            )
            with pytest.raises(ValueError, match='requires|reserved'):
                backend._validate_expert_tensor_parallel(model, invalid_plan.tensor_parallel_plan.plan)
        backend.prepare_model(model, plan)
        assert model.moe.experts is experts
        from torchturbo.engine.backends.fsdp2.prefetch import _find_experts_container
        assert _find_experts_container(model) is experts
        assert backend._parameter_ownership.is_expert(experts.weight1)
        assert backend._parameter_ownership.is_expert(experts.weight2)
        assert not backend._parameter_ownership.is_expert(model.moe.shared_expert.gate.weight)
        assert model.moe.config.moe_aux_loss_coeff == .02
        assert model.moe.config.moe_z_loss_coeff == .003
        assert experts.weight1.placements == (Shard(0), Shard(2))
        assert experts.weight2.placements == (Shard(0), Shard(1))
        if rank // 2 == 3:
            assert experts.weight1.to_local().numel() == 0
            assert experts.weight2.to_local().numel() == 0
        assert model.moe.router.routing_group is context.etp_mesh.get_group()
        assert model.moe.router.sp_group is context.sp_mesh.get_group()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=.001)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            inputs = [torch.arange(12.).view(2, 2, 3)/11 + i*.2 + step*.1 for i in range(world//2)]
            actual = model(inputs[rank//2])
            expected = reference(inputs[rank//2])
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            actual.square().sum().backward()
            ref_loss = sum(reference(value).square().sum() for value in inputs) / (world // 2)
            # The one-expert AUX objective is constant. Add Z directly so the
            # oracle uses ordinary scalar averaging, outside the custom scaler.
            ref_z_loss = sum(
                nn.functional.linear(value, reference.moe.router.weight).square().mean()
                for value in inputs
            ) * .003 / (world // 2)
            (ref_loss + ref_z_loss).backward()
            for name, parameter in model.named_parameters():
                expected_grad = reference.get_parameter(name).grad
                torch.testing.assert_close(parameter.grad.full_tensor().reshape_as(expected_grad), expected_grad, rtol=3e-5, atol=2e-6, msg=name)
            optimizer.step()
            ref_optimizer.step()
            for name, parameter in model.named_parameters():
                expected_value = reference.get_parameter(name)
                torch.testing.assert_close(parameter.full_tensor().reshape_as(expected_value), expected_value, rtol=3e-5, atol=2e-6, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('world', [4, 8])
def test_native_fsdp_etp_shared_expert_adamw(tmp_path, world):
    mp.spawn(_fsdp_worker, args=(str(tmp_path / 'store'), world), nprocs=world, join=True)


@pytest.mark.parametrize('contract', ['nongated', 'te', 'input_probs', 'features', 'wrong_module'])
def test_expert_style_rejects_unsupported_contract(contract):
    from types import SimpleNamespace
    from torchturbo.engine.tensor_parallel.expert import GroupedMLPParallel
    model = _experts()
    messages = dict(nongated='gated_linear_unit', te='use_te',
                    input_probs='moe_apply_probs_on_input', features='divisible',
                    wrong_module='shared GroupedMLP')
    if contract == 'nongated':
        model.config.gated_linear_unit = False
    elif contract == 'te':
        model.use_te = True
    elif contract == 'input_probs':
        model.config.moe_apply_probs_on_input = True
    elif contract == 'features':
        model.config.moe_ffn_hidden_size = 3
    else:
        model = nn.Linear(3, 4)
    with pytest.raises(ValueError, match=messages[contract]):
        GroupedMLPParallel()._apply(model, SimpleNamespace(size=lambda: 2))


@pytest.mark.parametrize('name,shape', [
    ('weight1', (8, 9)), ('weight1', (9, 7)),
    ('weight2', (3, 12)), ('weight2', (11, 3)),
])
def test_expert_style_checks_both_exact_packed_shapes_before_rebinding(monkeypatch, name, shape):
    from types import SimpleNamespace
    from torchturbo.engine.tensor_parallel.expert import GroupedMLPParallel
    model = _experts()
    model.register_parameter(name, nn.Parameter(torch.zeros(shape)))

    def reject_distribution(*args, **kwargs):
        pytest.fail("Expert distribution was reached before packed shape admission.")

    monkeypatch.setattr(
        "torchturbo.engine.tensor_parallel.expert.distribute_tensor",
        reject_distribution,
    )
    weight1, weight2 = model.weight1, model.weight2
    with pytest.raises(ValueError, match=name + '.*shape'):
        GroupedMLPParallel()._apply(model, SimpleNamespace(size=lambda: 2))
    assert model.weight1 is weight1
    assert model.weight2 is weight2
    assert not model._forward_pre_hooks
    assert not model._forward_hooks


@pytest.mark.parametrize('field', ['moe_ffn_hidden_size', 'hidden_size', 'num_local_experts'])
def test_expert_style_float_metadata_rejects_before_rebinding(field):
    from types import SimpleNamespace
    from torchturbo.engine.tensor_parallel.expert import GroupedMLPParallel
    model = _experts()
    owner = model if field == 'num_local_experts' else model.config
    setattr(owner, field, float(vars(owner)[field]))
    weight1, weight2 = model.weight1, model.weight2
    with pytest.raises(ValueError, match=field + '.*integer'):
        GroupedMLPParallel()._apply(model, SimpleNamespace(size=lambda: 2))
    assert model.weight1 is weight1
    assert model.weight2 is weight2
    assert not model._forward_pre_hooks
    assert not model._forward_hooks


def test_expert_style_keeps_empty_expert_count_supported():
    from types import SimpleNamespace
    from torchturbo.engine.tensor_parallel.expert import GroupedMLPParallel
    model = _experts(num_experts=0)
    style = GroupedMLPParallel()
    style.validate(model, 2)
    assert style._apply(model, SimpleNamespace(size=lambda: 1)) is model
    assert model.weight1.shape == (0, 8)
    assert model.weight2.shape == (0, 3)
