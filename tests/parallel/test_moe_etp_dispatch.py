"""Real CPU EP dispatch with aligned ETP and explicit deferred-op semantics.

Only the deferred permutation providers are substituted with CPU implementations;
this does not validate native NPU kernels or their stable ordering.
"""

from datetime import timedelta
from types import MethodType

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torchturbo.engine.tensor_parallel.expert import GroupedMLPParallel
from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext


class DeferredCPUDispatcher(MoEAlltoAllTokenDispatcher):
    @property
    def deferred_probs_to_unpermutation(self):
        return True


def _cpu_permutation(tokens, indices):
    order = indices.flatten().argsort(stable=True)
    return tokens.index_select(0, order // indices.size(1)), order


def _cpu_unpermutation(tokens, order, probs):
    restored = tokens.index_select(0, order.argsort()).view(*probs.shape, -1)
    return (restored * probs.unsqueeze(-1)).sum(1)


def _offset(block, lane, step):
    first = 0 if step == 0 else block % 4
    second = (first + 1) % 4
    alternative = (first + 2) % 4
    value = torch.full((4,), -.1)
    value[first] = .1
    value[second] = 1e-6 - lane * 3e-6
    value[alternative] = -1e-6 + lane * 3e-6
    return value


def _inputs(block, step):
    return (torch.arange(18.).view(2, 3, 3) / 19 + block*.13 + step*.2)


def _dense_moe(x, router_weight, w1, w2, block, lane, step, deferred):
    logits = nn.functional.linear(x, router_weight) + _offset(block, lane, step)
    canonical_logits = nn.functional.linear(x.detach(), router_weight.detach()) + _offset(block, 0, step)
    ids = canonical_logits.topk(2, dim=-1).indices
    probs = logits.gather(-1, ids).softmax(-1)
    if deferred:
        probs = probs.to(torch.bfloat16)
    outputs = []
    for expert in range(4):
        gate, up = (x @ w1[expert]).chunk(2, -1)
        outputs.append((nn.functional.silu(gate) * up) @ w2[expert])
    all_outputs = torch.stack(outputs, dim=-2)
    selected = all_outputs.gather(-2, ids[..., None].expand(*ids.shape, 3))
    return (selected * probs[..., None]).sum(-2)


def _worker(rank, path, world, sp_size, use_ac, deferred):
    from torchturbo.ops import get_or_create_ops
    from torchturbo.runtime.cpu import CpuRuntime
    from torchturbo.models.modules.moe import token_dispatcher as dispatcher_module
    dist.init_process_group('gloo', init_method=f'file://{path}', rank=rank,
                            world_size=world, timeout=timedelta(seconds=120))
    try:
        get_or_create_ops(device_runtime=CpuRuntime())
        if deferred:
            dispatcher_module.ops.moe_token_permutation = _cpu_permutation
            dispatcher_module.ops.moe_token_unpermutation = _cpu_unpermutation
        context = ParallelContext('cpu', sp_size=sp_size, ep_size=2, tp_size=2, etp_size=2)
        lane, block, owner = rank % 2, rank // 2, context.ep_mesh.get_local_rank()
        config = MoEConfig(hidden_size=3, moe_ffn_hidden_size=4, num_moe_experts=4,
            expert_model_parallel_size=2, gated_linear_unit=True,
            activation_func=nn.functional.silu, moe_router_topk=2,
            moe_token_dispatcher_type='alltoall')
        model = MoELayer(config, context.ep_mesh.get_group(),
            submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
            sp_group=context.sp_mesh.get_group())
        model.router.routing_group = context.etp_mesh.get_group()
        if deferred:
            model.token_dispatcher = DeferredCPUDispatcher(2, model.local_expert_indices,
                config, context.ep_mesh.get_group())
        full_w1 = (torch.arange(96.).view(4, 3, 8)/109-.4)
        full_w2 = (torch.arange(48.).view(4, 4, 3)/79-.3)
        router_weight = torch.tensor([[.03, .02, -.01]]).expand(4, 3).clone()
        with torch.no_grad():
            model.experts.weight1.copy_(full_w1[owner*2:(owner+1)*2].reshape(6, 8))
            model.experts.weight2.copy_(full_w2[owner*2:(owner+1)*2].reshape(8, 3))
            model.router.weight.copy_(router_weight)
        GroupedMLPParallel()._apply(model.experts, context.etp_mesh)
        traces = []
        probability_tensors = []
        def router_hook(module, inputs, output):
            probs, routing, logits = output
            if probs.requires_grad:
                probs.retain_grad()
                probability_tensors.append(probs)
            traces.append({'routing': routing.detach().clone()})
        model.router.register_forward_hook(router_hook)
        def expert_hook(module, inputs):
            dispatcher = model.token_dispatcher
            traces[-1].update(tokens=inputs[0].detach().clone(), counts=inputs[1].clone(),
                input_splits=torch.from_numpy(dispatcher.input_splits.copy()),
                output_splits=torch.from_numpy(dispatcher.output_splits.copy()))
        model.experts.register_forward_pre_hook(expert_hook)
        for step in range(2):
            model.zero_grad(set_to_none=True)
            traces.clear()
            probability_tensors.clear()
            def gating(router, x):
                return nn.functional.linear(x, router.weight) + _offset(block, lane, step)
            model.router.gating = MethodType(gating, model.router)
            x = _inputs(block, step).requires_grad_()
            independent = model.router.gating(x).topk(2, dim=-1).indices
            canonical = (nn.functional.linear(x, model.router.weight) + _offset(block, 0, step)).topk(2, dim=-1).indices
            if lane:
                assert not torch.equal(independent, canonical)
            with set_checkpoint_early_stop(False):
                actual = checkpoint(lambda value: model(value)[0], x, use_reentrant=False) if use_ac else model(x)[0]
            ref_x = x.detach().clone().requires_grad_()
            ref_router = router_weight.clone().requires_grad_()
            ref_w1 = full_w1.clone().requires_grad_()
            ref_w2 = full_w2.clone().requires_grad_()
            expected = _dense_moe(ref_x, ref_router, ref_w1, ref_w2, block, lane, step, deferred)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            factors = torch.linspace(.3, 1.3, actual.numel()).view_as(actual)
            (actual * factors).sum().backward()
            (expected * factors).sum().backward()
            torch.testing.assert_close(model.router.weight.grad, ref_router.grad, rtol=3e-5, atol=2e-6)
            torch.testing.assert_close(x.grad, ref_x.grad, rtol=3e-5, atol=2e-6)
            assert probability_tensors[0].grad is not None
            assert probability_tensors[0].grad.abs().sum() > 0
            ref_probs = probability_tensors[0].detach().clone().requires_grad_()
            normalized_probs = ref_probs / ref_probs.sum(-1, keepdim=True)
            normalized_probs = normalized_probs * traces[0]['routing']
            if deferred:
                normalized_probs = normalized_probs.to(torch.bfloat16)
            probability_output = torch.zeros_like(x).reshape(-1, 3)
            for expert in range(4):
                gate, up = (x.detach().reshape(-1, 3) @ full_w1[expert]).chunk(2, -1)
                expert_output = (nn.functional.silu(gate) * up) @ full_w2[expert]
                probability_output = probability_output + expert_output * normalized_probs[:, expert, None]
            (probability_output * factors.reshape(-1, 3)).sum().backward()
            torch.testing.assert_close(probability_tensors[0].grad, ref_probs.grad, rtol=3e-5, atol=2e-6)
            for trace in traces:
                for value in trace.values():
                    received = [torch.empty_like(value) for _ in range(2)]
                    dist.all_gather(received, value, group=context.etp_mesh.get_group())
                    torch.testing.assert_close(received[0], received[1], rtol=0, atol=0)
            assert len(traces) == (2 if use_ac else 1)
            if step == 0 and owner == 1:
                assert traces[0]['tokens'].shape == (0, 3)
                assert model.experts.weight1.grad is not None
                assert model.experts.weight2.grad is not None
            if step == 1:
                patterns = [None] * world
                dist.all_gather_object(patterns, traces[0]['routing'].tolist())
                assert patterns[0] != patterns[2]
            # EP computes the sum from its source blocks; no FSDP divisor is applied.
            ref_w1.grad = ref_w2.grad = None
            ep_blocks = [peer//2 for peer in dist.get_process_group_ranks(context.ep_mesh.get_group())]
            for source_block in ep_blocks:
                y = _dense_moe(_inputs(source_block, step), router_weight, ref_w1, ref_w2,
                               source_block, lane, step, deferred)
                (y * factors).sum().backward()
            torch.testing.assert_close(model.experts.weight1.grad.full_tensor(),
                ref_w1.grad[owner*2:(owner+1)*2].view(6, 2, 4), rtol=3e-5, atol=2e-6)
            torch.testing.assert_close(model.experts.weight2.grad.full_tensor(),
                ref_w2.grad[owner*2:(owner+1)*2], rtol=3e-5, atol=2e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('world,sp_size', [(4, 1), (8, 2)])
@pytest.mark.parametrize('use_ac', [False, True])
@pytest.mark.parametrize('deferred', [False, True])
def test_aligned_ep_etp_dispatch(tmp_path, world, sp_size, use_ac, deferred):
    mp.spawn(_worker, args=(str(tmp_path/'store'), world, sp_size, use_ac, deferred),
             nprocs=world, join=True)
