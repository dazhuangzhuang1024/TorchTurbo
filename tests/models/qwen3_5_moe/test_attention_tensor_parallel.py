"""Gated full-attention heads and shared-expert dense TP boundaries."""

import copy
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from torchturbo.models.qwen3_5_moe_text.parallel_plan import get_parallel_plan
from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext

from .tp_helpers import model_style, text_model


def _attention_oracle(module, x, positions):
    from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import apply_rotary_pos_emb

    q, gate = F.linear(x, module.q_proj.weight).reshape(*x.shape[:2], 4, 16).chunk(2, -1)
    q = module.q_norm(q).transpose(1, 2)
    k = module.k_norm(F.linear(x, module.k_proj.weight).reshape(*x.shape[:2], 2, 8)).transpose(1, 2)
    v = F.linear(x, module.v_proj.weight).reshape(*x.shape[:2], 2, 8).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *positions)
    output = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1),
                                           v.repeat_interleave(2, 1), is_causal=True)
    output = output.transpose(1, 2).reshape(*x.shape[:2], -1) * gate.flatten(2).sigmoid()
    return F.linear(output, module.o_proj.weight)


def _worker(rank, rendezvous):
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=4, timeout=timedelta(seconds=90))
    try:
        context = ParallelContext("cpu", sp_size=1, ep_size=1, tp_size=2)
        torch.manual_seed(31)
        model = text_model()
        layer = model.model.layers[1]
        for parameter in layer.parameters():
            with torch.no_grad():
                parameter.copy_(torch.randn_like(parameter) * .12)
        reference_attention = copy.deepcopy(layer.self_attn)
        reference_shared = copy.deepcopy(layer.mlp.shared_expert)
        reference_gate = copy.deepcopy(layer.mlp.shared_expert_gate)
        model_style(get_parallel_plan())._apply(model, context.tp_mesh)
        fully_shard(layer.self_attn, mesh=context.dense_fsdp_mesh)
        fully_shard(layer.mlp.shared_expert, mesh=context.dense_fsdp_mesh)
        fully_shard(layer.mlp.shared_expert_gate, mesh=context.dense_fsdp_mesh)
        inputs = [torch.sin(torch.arange(64).reshape(1, 4, 16) * .17 + lane * .6)
                  for lane in range(2)]
        target = torch.linspace(-.7, .9, 64).reshape(1, 4, 16)
        positions = (torch.ones(1, 4, 8), torch.zeros(1, 4, 8))
        causal_mask = torch.full((4, 4), float("-inf")).triu(1)
        x = inputs[rank // 2].clone().requires_grad_()
        actual, _ = layer.self_attn(x, position_embeddings=positions,
                                    attention_mask=causal_mask)
        shared_actual = layer.mlp.shared_expert(x) * layer.mlp.shared_expert_gate(x).sigmoid()
        expected = _attention_oracle(reference_attention, x.detach(), positions)
        shared_expected = reference_shared(x.detach()) * reference_gate(x.detach()).sigmoid()
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(shared_actual, shared_expected, rtol=3e-5, atol=3e-6)
        ((actual + shared_actual) * target).sum().backward()
        for data in inputs:
            rx = data.clone().requires_grad_()
            output = _attention_oracle(reference_attention, rx, positions)
            shared = reference_shared(rx) * reference_gate(rx).sigmoid()
            (((output + shared) * target).sum() / 2).backward()
            if torch.equal(data, inputs[rank // 2]):
                torch.testing.assert_close(x.grad, rx.grad * 2, rtol=3e-5, atol=3e-6)
        for actual_module, reference_module in (
            (layer.self_attn, reference_attention),
            (layer.mlp.shared_expert, reference_shared),
            (layer.mlp.shared_expert_gate, reference_gate),
        ):
            for name, parameter in actual_module.named_parameters():
                expected_grad = reference_module.get_parameter(name).grad
                assert expected_grad.abs().max() > 1e-8, name
                gradient = parameter.grad.full_tensor()
                torch.testing.assert_close(gradient, expected_grad, rtol=3e-5, atol=3e-6)
                if name == "q_proj.weight":
                    paired = expected_grad.reshape(4, 2, 8, 16)
                    assert paired[:, 0].abs().max() > 1e-5
                    assert paired[:, 1].abs().max() > 1e-5
    finally:
        dist.destroy_process_group()


def test_full_query_gate_and_shared_expert_gradients(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "attention-store"),), nprocs=4, join=True)
