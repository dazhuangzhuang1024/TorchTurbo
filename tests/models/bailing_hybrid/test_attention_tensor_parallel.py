"""Native MLA rotary/latent gradients and dense/shared TP on CPU/Gloo."""

import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor, Shard

from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext
from .tp_helpers import cpu_bailing_leaves, make_bailing_model, mla_oracle, mlp_oracle


def full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _worker(rank, directory, family, gate, bias, fsdp):
    from torch.distributed.fsdp import fully_shard
    from torchturbo.models.bailing_hybrid.tensor_parallel import BailingTensorParallel
    directory = Path(directory)
    world = 4 if fsdp else 2
    dist.init_process_group("gloo", init_method=(directory / "store").as_uri(),
                            rank=rank, world_size=world, timeout=timedelta(seconds=90))
    evidence = dict(rank=rank, family=family, gate=gate, bias=bias, fsdp=fsdp, steps=[])
    try:
        with cpu_bailing_leaves():
            context = ParallelContext("cpu", sp_size=1, ep_size=1, tp_size=2)
            model = make_bailing_model(family, layer_group_size=1,
                                       q_lora_rank=8 if gate else None,
                                       gated_attention_proj_granularity_type=gate,
                                       use_qkv_bias=bias, rope_interleave=family != "vl")
            modules = (model.model.layers[1].attention, model.model.layers[0].mlp,
                       model.model.layers[1].mlp.shared_experts)
            references = copy.deepcopy(modules)
            config_before = copy.deepcopy(model.model.config.to_dict())
            shapes_before = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
            BailingTensorParallel()._apply(model, context.tp_mesh)
            model.register_sp_attention_interface(dict(sp_group=None, sp_size=1, sp_rank=0))
            expected_config = config_before.copy()
            expected_config["use_cache"] = False
            assert model.model.config.to_dict() == expected_config
            assert model.model.config.use_cache is False
            assert shapes_before == {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
            for name in ("num_attention_heads", "num_key_value_heads", "qk_head_dim",
                         "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
                assert model.model.config.to_dict()[name] == config_before[name]
            assert not isinstance(modules[0].kv_a_proj_with_mqa.weight, DTensor)
            assert not isinstance(modules[0].kv_a_layernorm.weight, DTensor)
            if gate:
                assert not isinstance(modules[0].q_a_proj.weight, DTensor)
                assert not isinstance(modules[0].q_a_layernorm.weight, DTensor)
                assert modules[0].g_proj.weight.placements == (Shard(0),)
            for mlp in modules[1:]:
                assert mlp.gate_proj.weight.placements == (Shard(0),)
                assert mlp.up_proj.weight.placements == (Shard(0),)
                assert mlp.down_proj.weight.placements == (Shard(1),)
            assert model.model.config.num_attention_heads == 4
            assert model.model.config.num_key_value_heads == 1
            for key in ("hidden_size", "head_dim", "kv_lora_rank", "q_lora_rank"):
                assert model.model.config.to_dict()[key] == config_before[key]
            assert modules[0].num_heads == 2
            assert modules[0].kv_b_proj.weight.shape == (32, 8)
            assert modules[0].kv_b_proj.weight.placements == (Shard(0),)
            query = modules[0].q_proj if modules[0].q_lora_rank is None else modules[0].q_b_proj
            assert query.weight.placements == (Shard(0),)
            assert modules[0].dense.weight.placements == (Shard(1),)
            assert not isinstance(model.model.word_embeddings.weight, DTensor)
            if family == "vl":
                assert not any(isinstance(parameter, DTensor) for parameter in model.model.visual.parameters())
                assert not any(isinstance(parameter, DTensor) for parameter in model.linear_proj.parameters())
            if fsdp:
                for module in modules:
                    fully_shard(module, mesh=context.dense_fsdp_mesh)
            actual_optimizer = torch.optim.AdamW([p for m in modules for p in m.parameters()], lr=0.003)
            reference_optimizer = torch.optim.AdamW([p for m in references for p in m.parameters()], lr=0.003)
            boundaries = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
            angles = torch.arange(48).reshape(1, 6, 8).float() * 0.071
            positions = (angles.cos(), angles.sin())
            target = torch.sin(torch.arange(96).reshape(1, 6, 16).float() * 0.23) + 0.4
            for step in range(2):
                actual_optimizer.zero_grad()
                reference_optimizer.zero_grad()
                inputs = [torch.sin(torch.arange(96).reshape(1, 6, 16).float() * 0.17
                                    + lane * 0.6 + step * 0.2) for lane in range(2 if fsdp else 1)]
                lane = rank // 2 if fsdp else 0
                hidden = inputs[lane].clone().requires_grad_()
                actual_attention = modules[0](hidden, positions, None,
                                               cu_seq_lens_q=boundaries, cu_seq_lens_k=boundaries,
                                               max_length_q=3, max_length_k=3)[0]
                actual = actual_attention + modules[1](hidden) + modules[2](hidden)
                expected = mla_oracle(references[0], hidden.detach(), positions, boundaries)
                expected = expected + mlp_oracle(references[1], hidden.detach()) + mlp_oracle(references[2], hidden.detach())
                torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
                (actual * target).sum().backward()
                for index, data in enumerate(inputs):
                    ref_hidden = data.clone().requires_grad_()
                    ref_output = mla_oracle(references[0], ref_hidden, positions, boundaries)
                    ref_output = ref_output + mlp_oracle(references[1], ref_hidden) + mlp_oracle(references[2], ref_hidden)
                    ((ref_output * target).sum() / len(inputs)).backward()
                    if index == lane:
                        expected_input_gradient = ref_hidden.grad * len(inputs)
                errors = {}
                for name in ("weight", "bias") if bias else ("weight",):
                    expected_gradient = references[0].kv_a_proj_with_mqa.get_parameter(name).grad
                    gradient = full(modules[0].kv_a_proj_with_mqa.get_parameter(name).grad)
                    latent_end = references[0].kv_lora_rank
                    rotary_end = latent_end + references[0].qk_rope_head_dim
                    for label, rows in (("latent", slice(0, latent_end)),
                                        ("rotary", slice(latent_end, rotary_end))):
                        assert expected_gradient[rows].norm() > 0
                        torch.testing.assert_close(gradient[rows], expected_gradient[rows], rtol=5e-5, atol=5e-6)
                        errors[name + "." + label + "_norm"] = float(expected_gradient[rows].norm())
                torch.testing.assert_close(hidden.grad, expected_input_gradient, rtol=5e-5, atol=5e-6)
                for label, actual_module, reference in zip(("mla", "dense", "shared"), modules, references):
                    for name, parameter in actual_module.named_parameters():
                        expected_gradient = reference.get_parameter(name).grad
                        gradient = full(parameter.grad)
                        torch.testing.assert_close(gradient, expected_gradient, rtol=5e-5, atol=5e-6)
                        errors[label + "." + name] = float((gradient - expected_gradient).abs().max())
                        assert expected_gradient.norm() > 0, (label, name)
                actual_optimizer.step()
                reference_optimizer.step()
                for actual_module, reference in zip(modules, references):
                    for name, parameter in actual_module.named_parameters():
                        rp = reference.get_parameter(name)
                        torch.testing.assert_close(full(parameter), rp, rtol=5e-5, atol=5e-6)
                        for state in ("exp_avg", "exp_avg_sq"):
                            torch.testing.assert_close(full(actual_optimizer.state[parameter][state]),
                                                       reference_optimizer.state[rp][state], rtol=5e-5, atol=5e-6)
                evidence["steps"].append(errors)
            # Actual public roots/backbones run with default cache normalized by TP.
            if not fsdp:
                for kwargs in ({}, {"use_cache": None}, {"use_cache": False}):
                    output = model(input_ids=torch.tensor([[2, 3, 4, 5, 6, 7]]),
                                   cu_seq_lens_q=boundaries, cu_seq_lens_k=boundaries,
                                   max_length_q=3, max_length_k=3, **kwargs)
                    assert torch.isfinite(output.logits).all()
                for module in (model, model.model):
                    with pytest.raises(ValueError, match="cache"):
                        module(input_ids=torch.ones(1, 2, dtype=torch.long), use_cache=True)
    finally:
        (directory / f"rank-{rank}.json").write_text(json.dumps(evidence, indent=2))
        dist.destroy_process_group()


@pytest.mark.parametrize("family,gate,bias,fsdp", [
    ("hybrid", None, False, False),
    ("hybrid", "head_wise", True, False),
    ("vl", "element_wise", True, False),
    ("vl", "element_wise", True, True),
], ids=["hybrid-direct", "hybrid-lora-head", "vl-lora-element", "vl-fsdp"])
def test_native_mla_and_mlp_gradients(tmp_path, family, gate, bias, fsdp):
    mp.spawn(_worker, args=(str(tmp_path), family, gate, bias, fsdp),
             nprocs=4 if fsdp else 2, join=True)
