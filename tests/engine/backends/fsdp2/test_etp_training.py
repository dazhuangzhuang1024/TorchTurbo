"""Actual shared experts, native FSDP divisors, and checkpoint bookkeeping."""

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from torchturbo.engine.backends.fsdp2 import backend as fsdp_backend
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import EngineConfig, MixedPrecisionPolicy
from torchturbo.engine.optimizers import OptimizerClassPlan
from torchturbo.ops import get_or_create_ops
from torchturbo.parallel import FSDPParallelPlan, FSDPPlanSpec, FSDPPlacementSpec
from torchturbo.engine.tensor_parallel.expert import GroupedMLPParallel
from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext
from torchturbo.runtime.cpu import CpuRuntime
from tests.engine.backends.fsdp2.test_expert_tensor_parallel import SharedMoEModel, _reference


def _plan(context, *, checkpoint=False, second=False):
    names = ['moe', 'other_moe'] if second else ['moe']
    plan = FSDPParallelPlan(sp_size=context.sp_size, ep_size=context.ep_size,
                           tp_size=2, etp_size=2)
    for name in names:
        plan.plan[f'{name}.experts'] = FSDPPlanSpec(
            placement=FSDPPlacementSpec(Shard(0), 'edp'))
        if checkpoint:
            plan.plan[name] = FSDPPlanSpec(activation_checkpoint=True)
        plan.tensor_parallel_plan.plan.update({
            f'{name}.experts': GroupedMLPParallel(),
            f'{name}.shared_expert.gate': ColwiseParallel(),
            f'{name}.shared_expert.up': ColwiseParallel(),
            f'{name}.shared_expert.down': RowwiseParallel(input_layouts=Shard(-1)),
        })
    return plan


def _backend(context, *, bf16=False):
    config = EngineConfig(mixed_precision=(MixedPrecisionPolicy.BF16 if bf16
                                          else MixedPrecisionPolicy.FP32))
    backend = FSDP2Backend(config, CpuRuntime())
    backend._parallel_context = context
    backend._world_mesh = context.dense_fsdp_mesh
    backend._mesh = context.expert_mesh
    return backend


def _full(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def _write_rank(directory, rank, evidence):
    Path(directory, f'rank-{rank}.json').write_text(json.dumps(evidence, indent=2))


def _divisor_worker(rank, directory, world, ep_size, num_experts, dtype, bf16, mode, reported_version):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/store', rank=rank,
                            world_size=world, timeout=timedelta(seconds=120))
    try:
        get_or_create_ops(device_runtime=CpuRuntime())
        context = ParallelContext('cpu', sp_size=1, ep_size=ep_size, tp_size=2, etp_size=2)
        owner = context.ep_mesh.get_local_rank()
        q = context.expert_fsdp_mesh.size()
        k = context.dense_fsdp_mesh.size()
        model = SharedMoEModel(context.ep_mesh.get_group(),
            num_moe_experts=num_experts * ep_size).to(dtype)
        if mode == 'anchors':
            model.other_moe = SharedMoEModel(context.ep_mesh.get_group(),
                num_moe_experts=num_experts * ep_size).to(dtype).moe
        with torch.no_grad():
            model.moe.experts.weight1.add_(owner * .07)
            model.moe.experts.weight2.add_(owner * .11)
        w1 = nn.Parameter(model.moe.experts.weight1.detach().clone().view(num_experts, 3, 8))
        w2 = nn.Parameter(model.moe.experts.weight2.detach().clone().view(num_experts, 4, 3))
        calls, sum_calls = [], []
        real_divisor = FSDPModule.set_gradient_divide_factor
        real_sum = FSDPModule.set_force_sum_reduction_for_comms
        def divide(module, factor):
            calls.append((id(module), factor))
            return real_divisor(module, factor)
        def force_sum(module, enabled):
            sum_calls.append((id(module), enabled))
            return real_sum(module, enabled)
        FSDPModule.set_gradient_divide_factor = divide
        FSDPModule.set_force_sum_reduction_for_comms = force_sum
        backend = _backend(context, bf16=bf16)
        if mode == 'reject':
            reported_version = reported_version or fsdp_backend.torch_version()
            evidence = dict(dtype=str(dtype), q=q, k=k, runtime_version=torch.__version__,
                            reported_version=reported_version)
            _write_rank(directory, rank, evidence)
            # Spawned workers must override admission locally, not in the parent.
            with patch.object(fsdp_backend, 'torch_version', return_value=reported_version):
                with pytest.raises(ValueError, match='float16/float64 effective gradient reduction'):
                    backend.prepare_model(model, _plan(context))
            assert calls == []
            evidence.update(rejected=True, divisor_calls=len(calls))
            _write_rank(directory, rank, evidence)
            return
        # A backend reused after legacy EP preparation retains compensation.
        backend.gradient_scaling_factor = .5
        backend.prepare_model(model, _plan(context, second=mode == 'anchors'))
        assert backend.gradient_scaling_factor == 1.0
        experts = model.moe.experts
        if mode == 'anchors':
            assert sorted(calls) == sorted([(id(experts), k), (id(model.other_moe.experts), k)])
            _write_rank(directory, rank, dict(anchors=len(calls), q=q, k=k))
            return
        optimizer = backend.prepare_optimizer(OptimizerClassPlan(
            model, torch.optim.AdamW, {'lr': .003, 'eps': 1e-6, 'weight_decay': .02}))
        ref_optimizer = torch.optim.AdamW([w1, w2], lr=.003, eps=1e-6, weight_decay=.02)
        compute_dtype = torch.bfloat16 if bf16 else dtype
        tolerance = dict(rtol=.025, atol=3e-4) if bf16 or dtype == torch.float16 else dict(rtol=3e-5, atol=3e-6)
        evidence = dict(q=q, k=k, owner=owner, dtype=str(dtype), bf16=bf16,
                        reduction_dtype=str(torch.float32 if bf16 else dtype), steps=[])
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            counts = [2 + (expert + step) % 2 for expert in range(num_experts)]
            x = (torch.arange(sum(counts)*3, dtype=dtype).view(-1, 3)/23 + .2*step).requires_grad_()
            actual, _ = experts(x, torch.tensor(counts), torch.ones(sum(counts), dtype=dtype))
            expected = _reference(x.detach().to(compute_dtype), w1.to(compute_dtype),
                                  w2.to(compute_dtype), counts)
            torch.testing.assert_close(actual, expected, **tolerance)
            factors = torch.linspace(.2, .9, actual.numel(), dtype=compute_dtype).view_as(actual)
            coefficient = context.expert_fsdp_mesh.get_local_rank() + 1
            (actual * factors).sum().mul(coefficient).backward()
            scale = q * (q + 1) / (2 * k)
            (expected * factors).sum().mul(scale).backward()
            step_evidence = dict(expected_scale=scale, gradients={})
            for name, parameter, reference in [('weight1', experts.weight1, w1),
                                               ('weight2', experts.weight2, w2)]:
                assert parameter.placements == (Shard(0), Shard(2 if name == 'weight1' else 1))
                gradient = parameter.grad.full_tensor().reshape_as(reference)
                ratio = (gradient.double().flatten() @ reference.grad.double().flatten()) / reference.grad.double().square().sum()
                step_evidence['gradients'][name] = dict(ratio=float(ratio),
                    max_error=float((gradient-reference.grad).abs().max()),
                    local_shape=list(parameter.to_local().shape))
                torch.testing.assert_close(gradient, reference.grad, msg=f'{name}: observed/expected={ratio}', **tolerance)
                assert parameter.dtype == dtype
                assert parameter.grad.to_local().shape == parameter.to_local().shape
                rows = num_experts * 3 if name == 'weight1' else num_experts
                chunk = (rows + q - 1) // q
                local_rows = max(0, min(chunk, rows - context.expert_fsdp_mesh.get_local_rank()*chunk))
                assert parameter.to_local().shape[0] == local_rows
            optimizer.step()
            ref_optimizer.step()
            for parameter, reference in [(experts.weight1, w1), (experts.weight2, w2)]:
                torch.testing.assert_close(parameter.full_tensor().reshape_as(reference), reference, **tolerance)
                for state_name in ('exp_avg', 'exp_avg_sq', 'step'):
                    actual_state = _full(optimizer.state[parameter][state_name])
                    ref_state = ref_optimizer.state[reference][state_name]
                    torch.testing.assert_close(actual_state.reshape_as(ref_state), ref_state, **tolerance)
            evidence['steps'].append(step_evidence)
        assert calls == [(id(experts), k)]
        expected_sum = q > 1 and k > q and (bf16 or dtype in (torch.float32, torch.bfloat16))
        assert sum_calls == ([(id(experts), True)] if expected_sum else [])
        evidence['force_sum'] = expected_sum
        _write_rank(directory, rank, evidence)
    finally:
        dist.destroy_process_group()


def _run_divisor(tmp_path, world, ep_size, num_experts=1, dtype=torch.float32, bf16=False, mode='numeric', reported_version=None):
    mp.spawn(_divisor_worker, args=(str(tmp_path), world, ep_size, num_experts, dtype, bf16, mode, reported_version),
             nprocs=world, join=True)
    assert len(list(tmp_path.glob('rank-*.json'))) == world


def test_expert_backward_uses_logical_divisor_without_clipping(tmp_path):
    _run_divisor(tmp_path, 4, 2)


@pytest.mark.parametrize('world,ep_size', [(4, 1), (4, 2), (8, 2)])
def test_backend_configures_each_expert_fsdp_unit(tmp_path, world, ep_size):
    _run_divisor(tmp_path, world, ep_size, mode='anchors')


@pytest.mark.parametrize('world,num_experts', [(8, 1), (8, 3), (16, 1)])
def test_empty_expert_fsdp_shards_train_with_divisor(tmp_path, world, num_experts):
    _run_divisor(tmp_path, world, 2, num_experts)


@pytest.mark.parametrize('world,ep_size,dtype,bf16', [
    (4, 1, torch.float32, False),
    (4, 2, torch.float32, True),
    (4, 2, torch.float64, True),
    (8, 2, torch.float32, True),
    (8, 2, torch.float64, True),
    (16, 2, torch.float32, True),
    (8, 2, torch.float64, False),
])
def test_gloo_custom_divisor_uses_supported_public_policy(tmp_path, world, ep_size, dtype, bf16):
    _run_divisor(tmp_path, world, ep_size, dtype=dtype, bf16=bf16)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float64])
@pytest.mark.parametrize('reported_version', [None, (2, 11, 0), (2, 14, 1)],
                         ids=['installed', 'reported-2.11.0', 'reported-2.14.1'])
def test_q1_unsupported_effective_reductions_fail_before_training(tmp_path, dtype, reported_version):
    _run_divisor(tmp_path, 4, 2, dtype=dtype, mode='reject', reported_version=reported_version)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float64])
def test_q1_unit_divisor_preserves_effective_dtype(tmp_path, dtype):
    _run_divisor(tmp_path, 2, 1, dtype=dtype)


class SharedMoETrainingModel(SharedMoEModel):
    """Task4 shared-MoE fixture with explicit activation/mask SP boundaries."""

    def get_sequence_parallel_input_slicing_spec(self):
        from torchturbo.models.modules.attention.ulysses import SequenceParallelInputSlicingSpec
        return SequenceParallelInputSlicingSpec(self, type(self).forward,
                                                sliced_inputs=(('x', 0),), padded_inputs=())

    def get_moe_runtime_inputs(self, inputs, context):
        from torchturbo.models.modules.moe.moe_layer import MoERuntimeInputs
        sp = context['sp']
        mask = inputs['padding_mask'].chunk(sp['sp_size'], dim=0)[sp['sp_rank']]
        return MoERuntimeInputs(padding_mask=mask)


def _tensor_snapshot(model, optimizer=None):
    values = {}
    for name, parameter in model.named_parameters():
        if optimizer is None:
            assert parameter.grad is not None, name
            values[name] = parameter.grad.full_tensor().detach().clone()
        else:
            values[name] = parameter.full_tensor().detach().clone()
            for state_name in ('exp_avg', 'exp_avg_sq', 'step'):
                values[f'{name}/{state_name}'] = _full(optimizer.state[parameter][state_name]).detach().clone()
    return values


def _assert_snapshot(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=3e-5, atol=3e-6, msg=name)


def _logical_norm(gradients, ep_group):
    squares = torch.zeros(())
    for name, value in gradients.items():
        contribution = value.float().square().sum()
        if '.experts.' in name:
            dist.all_reduce(contribution, group=ep_group)
        squares += contribution
    return squares.sqrt()


def _ac_run(context, *, mode, clipping, early_metrics):
    from contextlib import contextmanager
    from torchturbo.engine.core import TorchTurboEngine
    from torchturbo.engine.metrics.tracker import MetricTracker
    from torchturbo.models.modules.moe.moe_utils import MoEAuxLossAutoScaler

    model = SharedMoETrainingModel(context.ep_mesh.get_group(), num_moe_experts=4,
        moe_router_topk=2, moe_router_enable_expert_bias=True)
    with torch.no_grad():
        model.moe.router.expert_bias.copy_(torch.tensor([0., .002, -.001, .001]))
        owner = context.ep_mesh.get_local_rank()
        model.moe.experts.weight1.add_(owner * .07)
        model.moe.experts.weight2.add_(owner * .11)
    backend = _backend(context)
    backend.config.moe.aux_loss_coeff = .05
    backend.config.moe.z_loss_coeff = .03
    backend.config.activation_checkpointing.enabled = mode != 'off'
    backend.config.activation_checkpointing.use_reentrant = mode == 'reentrant'
    backend.prepare_model(model, _plan(context, checkpoint=mode != 'off'))
    engine = TorchTurboEngine.__new__(TorchTurboEngine)
    engine.backend = backend
    engine.config = backend.config
    optimizer = backend.prepare_optimizer(OptimizerClassPlan(
        model, torch.optim.AdamW, {'lr': .001, 'eps': 1e-6, 'weight_decay': .02}))
    tracker = MetricTracker()
    router = model.moe.router
    probability_tensors, routing_maps, scales, prefetch_events = [], [], [], []
    def router_hook(module, inputs, output):
        probs, routing, _ = output
        if torch.is_grad_enabled():
            probs.retain_grad()
            probability_tensors.append(probs)
        routing_maps.append(routing.detach().clone())
    router.register_forward_hook(router_hook)
    real_backward = MoEAuxLossAutoScaler.backward
    def backward(ctx, grad_output):
        scales.append(float(MoEAuxLossAutoScaler.loss_scale))
        return real_backward(ctx, grad_output)
    MoEAuxLossAutoScaler.backward = staticmethod(backward)
    real_suspend = backend.prefetch_manager.suspend_expert_forward_prefetch
    @contextmanager
    def suspend(model):
        prefetch_events.append('enter')
        try:
            with real_suspend(model):
                yield
        finally:
            prefetch_events.append('exit')
    backend.prefetch_manager.suspend_expert_forward_prefetch = suspend
    result = dict(micros=[], steps=[], counters=[], metric_counts=[], metric_totals=[], scales=scales)
    expected_counts = torch.zeros(4)
    try:
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                probability_tensors.clear()
                routing_maps.clear()
                data_rank = context.data_dp_mesh.get_local_rank()
                x = (torch.arange(24.).view(4, 2, 3)/19 - .6 + data_rank*.11 + step*.13 + micro*.07).requires_grad_()
                mask = torch.zeros(4, 2, dtype=torch.bool)
                mask[-1, -1] = True
                engine._prepare_moe_runtime(model, {'padding_mask': mask}, 2)
                before = expected_counts.clone()
                output = model(x=x)
                local_mask = mask.chunk(context.sp_size, 0)[context.sp_mesh.get_local_rank()].reshape(-1)
                increment = routing_maps[0].masked_fill(local_mask[:, None], False).sum(0).float()
                expected_counts += increment
                if mode != 'reentrant':
                    torch.testing.assert_close(router.local_tokens_per_expert, expected_counts)
                elif step or micro:
                    torch.testing.assert_close(router.local_tokens_per_expert, before)
                if early_metrics:
                    tracker.collect(model)
                output.square().sum().div(2).backward()
                tracker.collect(model)
                assert tracker.counts == dict.fromkeys(('aux_loss', 'aux_loss_scaled', 'z_loss', 'z_loss_scaled'), step*2+micro+1)
                torch.testing.assert_close(router.local_tokens_per_expert, expected_counts,
                                           msg=f'{mode} counts after backward')
                assert router.pop_metrics() == {}
                assert len(scales) == (step*2+micro+1)*2
                assert all(scale == .5 for scale in scales)
                grads = [value.grad for value in probability_tensors if value.grad is not None]
                assert len(grads) == 1
                assert grads[0].abs().sum() > 0
                result['micros'].append(dict(output=output.detach().clone(),
                    input=x.grad.detach().clone(), probability=grads[0].detach().clone(),
                    gradients=_tensor_snapshot(model)))
                result['counters'].append(expected_counts.tolist())
                result['metric_counts'].append(dict(tracker.counts))
                result['metric_totals'].append({name: value.clone() for name, value in tracker.totals.items()})
                if mode != 'off':
                    assert prefetch_events == ['enter', 'exit'] * (step*2+micro+1)
            gradients = _tensor_snapshot(model)
            norm = _logical_norm(gradients, context.ep_mesh.get_group())
            if clipping:
                actual_norm = backend.clip_grad_norm_(model.parameters(), .03)
                torch.testing.assert_close(actual_norm, norm, rtol=3e-5, atol=3e-6)
                coefficient = (.03 / (norm + 1e-6)).clamp(max=1)
                clipped = _tensor_snapshot(model)
                _assert_snapshot(clipped, {name: value*coefficient for name, value in gradients.items()})
            optimizer.step()
            result['steps'].append(dict(norm=norm, values=_tensor_snapshot(model, optimizer)))
        torch.testing.assert_close(tracker.totals['aux_loss_scaled'], tracker.totals['aux_loss']*.05)
        torch.testing.assert_close(tracker.totals['z_loss_scaled'], tracker.totals['z_loss']*.03)
        torch.testing.assert_close(router.expert_bias, torch.tensor([0., .002, -.001, .001]))
        return result
    finally:
        MoEAuxLossAutoScaler.backward = real_backward


def _ac_worker(rank, directory, sp_size, mode, clipping, early_metrics):
    world = 4 * sp_size
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/store', rank=rank,
                            world_size=world, timeout=timedelta(seconds=120))
    try:
        get_or_create_ops(device_runtime=CpuRuntime())
        context = ParallelContext('cpu', sp_size=sp_size, ep_size=2, tp_size=2, etp_size=2)
        expected = _ac_run(context, mode='off', clipping=clipping, early_metrics=early_metrics)
        actual = _ac_run(context, mode=mode, clipping=clipping, early_metrics=early_metrics)
        if clipping:
            unclipped = _ac_run(context, mode='off', clipping=False, early_metrics=early_metrics)
            for left, right in zip(actual['micros'][:2], unclipped['micros'][:2]):
                _assert_snapshot(left['gradients'], right['gradients'])
            torch.testing.assert_close(actual['steps'][0]['norm'], unclipped['steps'][0]['norm'])
        assert actual['counters'] == expected['counters']
        assert actual['metric_counts'] == expected['metric_counts']
        assert actual['scales'] == expected['scales']
        for left, right in zip(actual['metric_totals'], expected['metric_totals']):
            _assert_snapshot(left, right)
        errors = dict(gradients=0., values=0., input=0., probability=0., output=0.)
        for left, right in zip(actual['micros'], expected['micros']):
            _assert_snapshot(left['gradients'], right['gradients'])
            errors['gradients'] = max(errors['gradients'], max(
                float((left['gradients'][name] - right['gradients'][name]).abs().max())
                for name in left['gradients']))
            for name in ('output', 'input', 'probability'):
                torch.testing.assert_close(left[name], right[name], rtol=3e-5, atol=3e-6, msg=name)
                errors[name] = max(errors[name], float((left[name]-right[name]).abs().max()))
        for left, right in zip(actual['steps'], expected['steps']):
            torch.testing.assert_close(left['norm'], right['norm'], rtol=3e-5, atol=3e-6)
            _assert_snapshot(left['values'], right['values'])
            errors['values'] = max(errors['values'], max(
                float((left['values'][name] - right['values'][name]).abs().max())
                for name in left['values']))
        _write_rank(directory, rank, dict(mode=mode, sp_size=sp_size, clipping=clipping,
            early_metrics=early_metrics, counters=actual['counters'],
            metrics=actual['metric_counts'], scales=actual['scales'], errors=errors,
            metric_totals=[{name: float(value) for name, value in totals.items()}
                           for totals in actual['metric_totals']],
            norms=[float(step['norm']) for step in actual['steps']]))
    finally:
        dist.destroy_process_group()


def _run_ac(tmp_path, sp_size, mode, clipping=False, early_metrics=False):
    mp.spawn(_ac_worker, args=(str(tmp_path), sp_size, mode, clipping, early_metrics),
             nprocs=4*sp_size, join=True)
    assert len(list(tmp_path.glob('rank-*.json'))) == 4*sp_size


@pytest.mark.parametrize('sp_size', [1, 2])
def test_nonreentrant_etp_ac_counts_logical_tokens_once(tmp_path, sp_size):
    _run_ac(tmp_path, sp_size, 'nonreentrant')


@pytest.mark.parametrize('sp_size', [1, 2])
def test_consumed_router_metrics_are_not_republished_by_etp_recompute(tmp_path, sp_size):
    _run_ac(tmp_path, sp_size, 'nonreentrant', early_metrics=True)


@pytest.mark.parametrize('sp_size', [1, 2])
@pytest.mark.parametrize('clipping', [False, True])
def test_reentrant_etp_ac_two_step_statistics_and_gradients(tmp_path, sp_size, clipping):
    _run_ac(tmp_path, sp_size, 'reentrant', clipping=clipping)


@pytest.mark.parametrize('sp_size', [1, 2])
def test_nonreentrant_etp_ac_clipping_matches_logical_norm(tmp_path, sp_size):
    _run_ac(tmp_path, sp_size, 'nonreentrant', clipping=True)


@pytest.fixture
def single_rank_group(tmp_path):
    dist.init_process_group('gloo', init_method=f'file://{tmp_path}/store',
                            rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('tp_size', [1, 2])
def test_etp_recompute_scope_restores_with_early_stop_and_exception(single_rank_group, tp_size):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from torchturbo.engine.model_fqn import build_model_fqn_map
    from torchturbo.models.modules.moe.moe_config import MoEConfig
    from torchturbo.models.modules.moe.moe_utils import router_statistics_enabled
    from torchturbo.models.modules.moe.router import TopKRouter

    class RouterBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = TopKRouter(MoEConfig(hidden_size=3, num_moe_experts=4,
                moe_router_topk=2, moe_router_score_function='sigmoid',
                moe_router_enable_expert_bias=True, moe_aux_loss_coeff=.05,
                moe_z_loss_coeff=.03))
            self.calls = 0
            self.completions = 0
            self.fail_on_call = None
            self.scopes = []

        def forward(self, x):
            self.calls += 1
            self.scopes.append(router_statistics_enabled())
            if self.calls == self.fail_on_call:
                raise RuntimeError('controlled replay failure')
            probabilities, _, _ = self.router(x)
            result = probabilities.sin().square()
            self.completions += 1
            return result

    model = nn.Module()
    model.block = RouterBlock()
    backend = FSDP2Backend(EngineConfig(), CpuRuntime())
    backend.config.activation_checkpointing.enabled = True
    backend._parallel_context = SimpleNamespace(tp_size=tp_size)
    events = []
    original_suspend = backend.prefetch_manager.suspend_expert_forward_prefetch
    @contextmanager
    def suspend(root):
        events.append('enter')
        try:
            with original_suspend(root):
                yield
        finally:
            events.append('exit')
    backend.prefetch_manager.suspend_expert_forward_prefetch = suspend
    plan = FSDPParallelPlan(plan={'block': FSDPPlanSpec(activation_checkpoint=True)})
    resolved = backend._compile_parallel_plan(plan, build_model_fqn_map(model))
    backend._apply_activation_checkpointing(model, resolved)
    x = torch.linspace(-.3, .7, 12).view(2, 2, 3).requires_grad_()
    model.block(x).sum().backward()
    replay_publishes = tp_size == 1
    assert model.block.scopes == [True, replay_publishes]
    assert model.block.completions == 1
    assert model.block.router.local_tokens_per_expert.sum() == (16 if replay_publishes else 8)
    assert events == ['enter', 'exit']
    assert router_statistics_enabled()
    model.block.fail_on_call = model.block.calls + 2
    value = model.block(x)
    with pytest.raises(RuntimeError, match='controlled replay failure'):
        value.sum().backward()
    assert model.block.scopes == [True, replay_publishes, True, replay_publishes]
    assert events == ['enter', 'exit', 'enter', 'exit']
    assert router_statistics_enabled()
    router = model.block.router
    before = router.local_tokens_per_expert.clone()
    router.pop_metrics()
    _, routing, _ = router(x)
    torch.testing.assert_close(router.local_tokens_per_expert, before + routing.sum(0))
    assert set(router.pop_metrics()) == {'aux_loss', 'aux_loss_scaled', 'z_loss', 'z_loss_scaled'}
