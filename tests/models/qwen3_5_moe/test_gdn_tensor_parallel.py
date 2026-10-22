"""Numerical local GDN boundaries, including replicated Partial SUM gradients."""

import copy
import importlib
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.nn import functional as F

from torchturbo.models.qwen3_5_moe_text.parallel_plan import get_parallel_plan
from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext

from .tp_helpers import (
    gdn_oracle, initialize_gdn, install_cpu_leaves, model_style, model_validator, text_model,
)


def test_cpu_leaf_bridge_delegates_grouped_matmul_to_real_dispatcher(monkeypatch):
    from torchturbo import ops as turbo_ops
    from torchturbo.runtime.cpu import CpuRuntime

    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    turbo_ops.get_or_create_ops(device_runtime=CpuRuntime())
    install_cpu_leaves(monkeypatch)
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    weights = torch.tensor([[[2.], [1.]], [[1.], [3.]]])
    groups = torch.tensor([2, 3], dtype=torch.int64)
    actual = turbo_ops.ops.grouped_matmul(x, weights, groups)
    torch.testing.assert_close(actual, torch.tensor([[4.], [10.], [23.]]), rtol=0, atol=0)


def test_bf16_cpu_conv_leaf_matches_widened_literal_accumulation(monkeypatch):
    """Pin the measured single-rounding CPU-build contract, not all backends."""
    from torchturbo import ops as turbo_ops

    install_cpu_leaves(monkeypatch)
    x = torch.linspace(-.8, .9, 24).reshape(1, 6, 4).to(torch.bfloat16).requires_grad_()
    weight = torch.cos(torch.arange(12) * .3).reshape(4, 3).to(torch.bfloat16).requires_grad_()
    reference_x, reference_weight = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()
    bounds = (0, 1, 4, 6)
    actual, _ = turbo_ops.ops.causal_conv1d(x, weight, None, activation="silu", cu_seqlens=torch.tensor(bounds, dtype=torch.int32))
    channels = []
    for channel in range(4):
        tokens = []
        for start, end in zip(bounds, bounds[1:]):
            for token in range(start, end):
                terms = [reference_x[:, source, channel].float() * reference_weight[channel, offset].float()
                    for offset in range(3) if (source := token - 2 + offset) >= start]
                tokens.append(F.silu(torch.stack(terms).sum(0).to(x.dtype)))
        channels.append(torch.stack(tokens, 1))
    expected = torch.stack(channels, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    target = torch.linspace(-.7, .8, 24).reshape_as(actual).to(x.dtype)
    (actual * target).sum().backward()
    (expected * target).sum().backward()
    for left, right in ((x.grad, reference_x.grad), (weight.grad, reference_weight.grad)):
        assert right.norm() > 0
        assert (left.float() - right.float()).norm() / right.float().norm() < .01


class SourceContractFusedNorm(torch.nn.Module):
    """Source-inspected FLA arithmetic; this is not an accelerator kernel."""

    def __init__(self, features, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(.7, 1.3, features))
        self.bias = None
        self.eps = eps
        self.activation = "silu"

    def forward(self, core, gate):
        normalized = core.float() * torch.rsqrt(core.float().square().mean(-1, keepdim=True) + self.eps)
        affine = normalized * self.weight.float()
        return (affine * gate.float() * gate.float().sigmoid()).to(core.dtype)


def _full(tensor, placements=None):
    if isinstance(tensor, DTensor):
        if placements is not None and tensor.placements != placements:
            tensor = tensor.redistribute(placements=placements)
        return tensor.full_tensor()
    return tensor


def test_cpu_gdn_bridge_matches_independent_recurrence_and_all_gradients(monkeypatch):
    install_cpu_leaves(monkeypatch)
    torch.manual_seed(11)
    model = text_model()
    module = model.model.layers[0].linear_attn
    initialize_gdn(module)
    reference = copy.deepcopy(module)
    x = torch.randn(1, 6, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    boundaries = (0, 1, 4, 6)
    actual = module(x, cu_seq_lens=torch.tensor(boundaries, dtype=torch.int32))
    expected = gdn_oracle(reference, reference_x, boundaries)
    target = torch.linspace(-.7, .9, actual.numel()).reshape_as(actual)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    (actual * target).sum().backward()
    (expected * target).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=3e-6, rtol=3e-5)
    for (name, parameter), (_, reference_parameter) in zip(
        module.named_parameters(), reference.named_parameters()
    ):
        assert reference_parameter.grad.abs().max() > 1e-8, name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad,
                                   atol=3e-6, rtol=3e-5, msg=name)


def test_cancellation_sensitive_rowwise_projection_preserves_gradient_vector_scale():
    module = text_model().model.layers[0].linear_attn
    initialize_gdn(module, cancellation_sensitive=True)
    reference = copy.deepcopy(module)
    x = torch.sin(torch.arange(96).reshape(1, 6, 16) * .13)
    target = torch.linspace(-.7, .9, 96).reshape_as(x)

    def rowwise(x, weight):
        return (torch.nn.functional.linear(x[..., :8], weight[:, :8])
                + torch.nn.functional.linear(x[..., 8:], weight[:, 8:]))

    actual = gdn_oracle(module, x, (0, 1, 4, 6), qkv_projection=rowwise)
    expected = gdn_oracle(reference, x, (0, 1, 4, 6))
    (actual * target).sum().backward()
    (expected * target).sum().backward()
    for (name, parameter), (_, ref) in zip(module.named_parameters(), reference.named_parameters()):
        assert ref.grad.norm() > 1e-7, name
        error = (parameter.grad - ref.grad).norm() / ref.grad.norm()
        scale = (parameter.grad.flatten() @ ref.grad.flatten()) / ref.grad.square().sum()
        assert error < 1e-5, (name, error)
        assert .995 < scale < 1.005, (name, scale)


def _worker(rank, rendezvous, fsdp, sp_size=1, key_heads=4, checkpoint=False):
    world_size = 4 if fsdp or sp_size > 1 else 2
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=90))
    monkeypatch = pytest.MonkeyPatch()
    try:
        install_cpu_leaves(monkeypatch)
        context = ParallelContext("cpu", sp_size=sp_size, ep_size=1, tp_size=2)
        torch.manual_seed(17)
        model = text_model(linear_num_key_heads=key_heads)
        module = model.model.layers[0].linear_attn
        initialize_gdn(module)
        reference = copy.deepcopy(module)
        original_shapes = {name: tuple(parameter.shape)
                           for name, parameter in module.named_parameters()}
        style = model_style(get_parallel_plan())
        validator, plan = model_validator(get_parallel_plan(), sp_size=sp_size)
        validator.validate(model, plan)
        style._apply(model, context.tp_mesh)
        # Public cache admission must happen before any projection or embedding.
        def forbidden(*args, **kwargs):
            raise AssertionError("Cache rejection executed model computation")
        handle = model.model.embed_tokens.register_forward_pre_hook(forbidden)
        for entry in (model, model.model):
            with pytest.raises(ValueError, match="cache|Cache"):
                entry(input_ids=torch.ones(1, 2, dtype=torch.long), use_cache=True)
            with pytest.raises(ValueError, match="cache|Cache"):
                entry(input_ids=torch.ones(1, 2, dtype=torch.long), past_key_values=object())
        handle.remove()
        assert module.num_k_heads == key_heads // 2
        assert module.num_v_heads == 4
        assert module.key_dim == key_heads * 2 and module.value_dim == 16
        assert model.config.linear_num_key_heads == key_heads
        assert model.config.linear_num_value_heads == 8
        for name, parameter in module.named_parameters():
            assert tuple(parameter.shape) == original_shapes[name], name
        assert module.in_proj_qkv.weight.placements == (Shard(1),)
        assert module.conv1d.weight.placements == (Replicate(),)
        assert module.norm.weight.placements == (Replicate(),)
        assert module.A_log.placements == (Shard(0),)
        assert module.dt_bias.placements == (Shard(0),)
        if fsdp:
            from torch.distributed.fsdp import fully_shard
            fully_shard(module, mesh=context.dense_fsdp_mesh, reshard_after_forward=True)
        if sp_size > 1:
            from torchturbo.models.qwen3_5_moe.model import TurboQwen3_5MoeForConditionalGeneration
            info = context.get_sp_info()
            TurboQwen3_5MoeForConditionalGeneration.patch_gated_delta_net_for_sp(
                module, info["sp_group"], sp_size, info["sp_rank"]
            )

        # Distinct data replicas must exercise DP mean after TP SUM restoration.
        inputs = [torch.sin(torch.arange(96).reshape(1, 6, 16) * .13 + lane * .7)
                  for lane in range(world_size // (2 * sp_size))]
        target = torch.linspace(-.7, .9, 96).reshape(1, 6, 16)
        boundaries = (0, 1, 4, 6)
        optimizer = torch.optim.AdamW(module.parameters(), lr=.003, weight_decay=.02)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.003, weight_decay=.02)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            data_rank = context.get_dp_info()["dp_rank"]
            full_x = inputs[data_rank] + step * .08
            start = context.get_sp_info()["sp_rank"] * (6 // sp_size)
            x = full_x[:, start:start + 6 // sp_size].clone().requires_grad_()
            offsets = torch.tensor(boundaries, dtype=torch.int32)
            if checkpoint:
                from torch.utils.checkpoint import checkpoint as checkpoint_forward
                actual = checkpoint_forward(module, x, cu_seq_lens=offsets,
                                            use_reentrant=False)
            else:
                actual = module(x, cu_seq_lens=offsets)
            expected = gdn_oracle(reference, full_x, boundaries)[:, start:start + 6 // sp_size]
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
            local_target = target[:, start:start + 6 // sp_size]
            ((actual * local_target).sum() * sp_size).backward()
            for data in inputs:
                reference_x = (data + step * .08).requires_grad_()
                oracle_output = gdn_oracle(reference, reference_x, boundaries)
                ((oracle_output * target).sum() / len(inputs)).backward()
                if torch.equal(data, inputs[data_rank]):
                    expected_grad = reference_x.grad[:, start:start + 6 // sp_size] * len(inputs) * sp_size
                    torch.testing.assert_close(x.grad, expected_grad,
                                               atol=3e-6, rtol=3e-5)
            for (name, parameter), (_, ref_parameter) in zip(
                module.named_parameters(), reference.named_parameters()
            ):
                assert ref_parameter.grad.abs().max() > 1e-8, name
                if not fsdp and name in ("conv1d.weight", "norm.weight"):
                    assert parameter.grad.placements == (Partial(),), name
                gradient = _full(parameter.grad, parameter.placements)
                torch.testing.assert_close(gradient, ref_parameter.grad,
                                           atol=3e-6, rtol=3e-5)
                factor = (gradient.flatten() @ ref_parameter.grad.flatten()) / ref_parameter.grad.square().sum()
                assert .995 < factor < 1.005, (name, factor)
                if not fsdp and parameter.grad.placements != parameter.placements:
                    parameter.grad = parameter.grad.redistribute(placements=parameter.placements)
            optimizer.step()
            reference_optimizer.step()
            for (name, parameter), (_, ref_parameter) in zip(
                module.named_parameters(), reference.named_parameters()
            ):
                torch.testing.assert_close(_full(parameter), ref_parameter,
                                           atol=3e-6, rtol=3e-5, msg=name)
                for state in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(_full(optimizer.state[parameter][state]),
                                               reference_optimizer.state[ref_parameter][state],
                                               atol=3e-6, rtol=3e-5, msg=f"{name}.{state}")
    finally:
        monkeypatch.undo()
        dist.destroy_process_group()


@pytest.mark.parametrize("fsdp", [False, True], ids=["tp", "tp-fsdp-partial-sum"])
def test_gdn_parameter_gradients_and_adamw_match_full_reference(tmp_path, fsdp):
    mp.spawn(_worker, args=(str(tmp_path / "store"), fsdp),
             nprocs=4 if fsdp else 2, join=True)


@pytest.mark.parametrize("key_heads", [2, 4, 8], ids=["sp-greater", "sp-equal", "sp-smaller"])
def test_gdn_tp_sp_repeat_and_fsdp_gradients(tmp_path, key_heads):
    mp.spawn(_worker, args=(str(tmp_path / "sp-store"), True, 2, key_heads),
             nprocs=4, join=True)


def test_gdn_local_norm_functional_call_survives_nonreentrant_recompute(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "ac-store"), True, 2, 2, True),
             nprocs=4, join=True)


def _norm_worker(rank, rendezvous):
    from torchturbo.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling
    from torchturbo.models.qwen3_5_moe.tensor_parallel import gated_norm

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    monkeypatch = pytest.MonkeyPatch()
    try:
        context = ParallelContext("cpu", sp_size=1, ep_size=1, tp_size=2)
        for fused, dtype in ((False, torch.float32), (False, torch.bfloat16),
                             (True, torch.bfloat16)):
            model = text_model()
            attention = model.model.layers[0].linear_attn
            if fused:
                monkeypatch.setattr(modeling, "FusedRMSNormGated", SourceContractFusedNorm)
                attention.norm = SourceContractFusedNorm(4, attention.layer_norm_epsilon)
            attention.norm.to(dtype=dtype)
            reference = copy.deepcopy(attention.norm)
            style = model_style(get_parallel_plan())
            style._apply(model, context.tp_mesh)
            core = torch.linspace(-1.3, 1.7, 96).reshape(1, 3, 8, 4).to(dtype)
            gate = torch.cos(torch.arange(96) * .11).reshape_as(core).to(dtype)
            target = torch.linspace(-.9, .8, 96).reshape_as(core).to(dtype)
            reference_core = core.clone().requires_grad_()
            reference_gate = gate.clone().requires_grad_()
            expected = reference(reference_core, reference_gate)
            (expected * target).sum().backward()
            local_core = core[:, :, rank * 4:(rank + 1) * 4].reshape(-1, 4).clone().requires_grad_()
            local_gate = gate[:, :, rank * 4:(rank + 1) * 4].reshape(-1, 4).clone().requires_grad_()
            original_weight = attention.norm.weight
            actual = gated_norm(attention, local_core, local_gate)
            assert attention.norm.weight is original_weight
            expected_local = expected[:, :, rank * 4:(rank + 1) * 4].reshape(-1, 4)
            torch.testing.assert_close(actual, expected_local, atol=0, rtol=0)
            local_target = target[:, :, rank * 4:(rank + 1) * 4].reshape(-1, 4)
            (actual * local_target).sum().backward()
            for actual_grad, full_grad in ((local_core.grad, reference_core.grad),
                                           (local_gate.grad, reference_gate.grad)):
                assert full_grad.abs().max() > .01
                expected_grad = full_grad[:, :, rank * 4:(rank + 1) * 4].reshape(-1, 4)
                torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
            gradient = attention.norm.weight.grad
            assert gradient.placements == (Partial(),)
            completed = gradient.redistribute(placements=[Replicate()]).to_local()
            # BF16 Partial contributions round locally before the sum.
            tolerance = .02 if dtype == torch.bfloat16 else 1e-6
            torch.testing.assert_close(completed, reference.weight.grad,
                                       atol=tolerance, rtol=tolerance)
            scale = (completed.float() @ reference.weight.grad.float()) / reference.weight.grad.float().square().sum()
            assert .98 < scale < 1.02
            def failing_forward(core, gate):
                assert not isinstance(attention.norm.weight, DTensor)
                raise RuntimeError("norm failure")
            with monkeypatch.context() as scoped:
                scoped.setattr(attention.norm, "forward", failing_forward)
                with pytest.raises(RuntimeError, match="norm failure"):
                    gated_norm(attention, local_core, local_gate)
            assert attention.norm.weight is original_weight
    finally:
        monkeypatch.undo()
        dist.destroy_process_group()


def test_gated_norm_preserves_both_input_gradients_and_selected_cast_contract(tmp_path):
    mp.spawn(_norm_worker, args=(str(tmp_path / "norm-store"),), nprocs=2, join=True)
