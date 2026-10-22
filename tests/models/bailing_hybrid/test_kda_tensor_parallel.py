"""Instance-owned KDA and live native TP/SP/FSDP boundaries on CPU/Gloo."""

import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext
from .tp_helpers import (cpu_bailing_leaves, cpu_chunk_kda, kda_oracle,
                         make_bailing_config, make_bailing_model)


@pytest.fixture(autouse=True)
def restore_attention_registry(monkeypatch):
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    for registry in (ALL_MASK_ATTENTION_FUNCTIONS, ALL_ATTENTION_FUNCTIONS):
        monkeypatch.setattr(type(registry), "_global_mapping", registry._global_mapping.copy())
        monkeypatch.setattr(registry, "_local_mapping", registry._local_mapping.copy())


def full(tensor):
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


@pytest.mark.parametrize("family", ["hybrid", "vl"])
def test_instance_owned_wrapper_shared_config_aba(monkeypatch, family):
    from torchturbo.engine.config import EngineConfig
    from torchturbo.models.modules.attention.ulysses import ulysses
    from torchturbo.models.bailing_hybrid.model import TurboBailingMoeV3ForCausalLM
    from torchturbo.models.bailing_moe_v3_vl.model import TurboBailingMoeV3VLForConditionalGeneration
    cls = (TurboBailingMoeV3ForCausalLM if family == "hybrid"
           else TurboBailingMoeV3VLForConditionalGeneration)
    groups = [object(), object()]
    ranks = {groups[0]: 0, groups[1]: 1}
    calls = []
    def gather_sequence(x, seq_dim, head_dim, sp_group, sp_size):
        calls.append(sp_group)
        return torch.cat([x] * sp_size, dim=seq_dim).chunk(sp_size, dim=head_dim)[ranks[sp_group]]
    def gather_heads(x, seq_dim, head_dim, sp_group, sp_size):
        calls.append(sp_group)
        return torch.cat([x] * sp_size, dim=head_dim).chunk(sp_size, dim=seq_dim)[ranks[sp_group]]
    monkeypatch.setattr(ulysses, "gather_seq_scatter_heads", gather_sequence)
    monkeypatch.setattr(ulysses, "gather_heads_scatter_seq", gather_heads)
    with cpu_bailing_leaves():
        config = make_bailing_config(family, use_cache=False)
        models = []
        x = torch.sin(torch.arange(48).reshape(1, 3, 16).float() * .17)
        boundaries = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
        for group in groups:
            with cls.init_context(config, EngineConfig()):
                model = cls(config)
                if not models:
                    attention_type = type(model.model.layers[0].attention)
                    monkeypatch.setattr(attention_type, "forward", attention_type.forward)
                    dense_forward = attention_type.forward
                    dense = copy.deepcopy(model.model.layers[0].attention)
                    with torch.no_grad():
                        dense.dt_bias.zero_()
                    dense_before = dense(x)[0]
                with torch.no_grad():
                    model.model.layers[0].attention.dt_bias.zero_()
                model.on_parallelized({"sp": dict(sp_group=group, sp_size=2, sp_rank=ranks[group])})
            models.append(model)
            if len(models) == 1:
                before = model.model.layers[0].attention(x, cu_seqlens=boundaries)[0]
            else:
                calls.clear()
                model.model.layers[0].attention(x, cu_seqlens=boundaries)
                assert calls and all(group is groups[1] for group in calls)
        assert models[0].config is models[1].config
        if family == "hybrid":
            with pytest.raises(NotImplementedError, match="flash_attention_2"):
                models[0].on_parallelized({"sp": dict(sp_group=groups[0], sp_size=2, sp_rank=0)})
        calls.clear()
        after = models[0].model.layers[0].attention(x, cu_seqlens=boundaries)[0]
        assert calls and all(group is groups[0] for group in calls)
        torch.testing.assert_close(after, before)
        assert attention_type.forward is dense_forward
        assert dense.forward.__func__ is dense_forward
        torch.testing.assert_close(dense(x)[0], dense_before)
        # The actual MLA modules resolve A/B/A through the shared registry.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        mla_calls = []
        def mla_leaf(query, key, value, mask, length, **kwargs):
            group = kwargs["sp_group"]
            mla_calls.append(group)
            return query * (ranks[group] + 1)
        monkeypatch.setattr(ulysses, "ulysses_flash_attention_forward", mla_leaf)
        query = torch.ones(1, 4, 3, 4)
        for model, group in zip((models[0], models[1], models[0]), (groups[0], groups[1], groups[0])):
            mla = model.model.layers[1].attention
            entry = ALL_ATTENTION_FUNCTIONS[mla.config._attn_implementation]
            result, _ = entry(mla, query, query, query, None)
            torch.testing.assert_close(result, query.transpose(1, 2) * (ranks[group] + 1))
        assert mla_calls == [groups[0], groups[1], groups[0]]
        # Explicit class compatibility cannot override a prepared instance.
        from torchturbo.models.bailing_hybrid.attention import patch_kda_forward
        patch_kda_forward(attention_type, groups[1], 2, 1)
        calls.clear()
        torch.testing.assert_close(models[0].model.layers[0].attention(x, cu_seqlens=boundaries)[0], before)
        assert all(group is groups[0] for group in calls)


def _worker(rank, directory, family, direct, safe_gate, lower_bound, sp_size, ac):
    from torch.distributed.fsdp import fully_shard
    from torch.utils.checkpoint import checkpoint
    from torchturbo.models.bailing_hybrid.tensor_parallel import BailingTensorParallel
    directory = Path(directory)
    world = 4 if sp_size == 2 else 2
    dist.init_process_group("gloo", init_method=(directory / "store").as_uri(),
                            rank=rank, world_size=world, timeout=timedelta(seconds=90))
    evidence = dict(rank=rank, family=family, direct=direct, safe_gate=safe_gate,
                    lower_bound=lower_bound, sp_size=sp_size, ac=ac, steps=[])
    try:
        with cpu_bailing_leaves():
            context = ParallelContext("cpu", sp_size=sp_size, ep_size=1, tp_size=2)
            model = make_bailing_model(family, no_kda_lora=direct,
                                       kda_safe_gate=safe_gate, kda_lower_bound=lower_bound)
            module = model.model.layers[0].attention
            reference = copy.deepcopy(module)
            config_before = copy.deepcopy(model.model.config.to_dict())
            shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
            style = BailingTensorParallel()
            from torchturbo.models.bailing_hybrid.parallel_plan import get_parallel_plan
            plan = get_parallel_plan()
            plan.tp_size = plan.etp_size = 2
            plan.sp_size = sp_size
            plan.tensor_parallel_plan.validator.validate(model, plan)
            style._apply(model, context.tp_mesh)
            model.on_parallelized({"sp": context.get_sp_info()})
            assert model.model.config.to_dict() == {**config_before, "use_cache": False}
            assert shapes == {name: tuple(p.shape) for name, p in model.named_parameters()}
            assert module.num_heads == module.num_k_heads == 2
            assert module.head_dim == module.head_k_dim == 4
            assert module.o_norm.weight.placements == (Replicate(),)
            for name in ("q_proj", "k_proj", "v_proj", "b_proj"):
                assert module.get_submodule(name).weight.placements == (Shard(0),)
            for name in (("f_proj", "g_proj") if direct else ("f_b_proj", "g_b_proj")):
                assert module.get_submodule(name).weight.placements == (Shard(0),)
            if not direct:
                assert not isinstance(module.f_a_proj.weight, DTensor)
                assert not isinstance(module.g_a_proj.weight, DTensor)
            for name in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "A_log", "dt_bias"):
                assert module.get_parameter(name).placements == (Shard(0),)
            assert module.o_proj.weight.placements == (Shard(1),)
            assert not isinstance(model.model.word_embeddings.weight, DTensor)
            if family == "vl":
                assert not any(isinstance(p, DTensor) for p in model.model.visual.parameters())
                assert not any(isinstance(p, DTensor) for p in model.linear_proj.parameters())
            if sp_size == 2:
                fully_shard(module, mesh=context.dense_fsdp_mesh, reshard_after_forward=True)
            norm_parameter = module.o_norm.weight
            norm_forward = module.o_norm.forward.__func__
            identities = []
            def record_norm(module, args):
                identities.append(module.o_norm.weight)
            handle = module.register_forward_pre_hook(record_norm)
            optimizer = torch.optim.AdamW(module.parameters(), lr=.003, weight_decay=.02)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.003, weight_decay=.02)
            boundaries = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
            target = torch.linspace(-.7, .9, 96).reshape(1, 6, 16)
            start = context.get_sp_info()["sp_rank"] * (6 // sp_size)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                full_x = torch.sin(torch.arange(96).reshape(1, 6, 16).float() * .17 + step * .08)
                x = full_x[:, start:start + 6 // sp_size].clone().requires_grad_()
                if ac:
                    actual = checkpoint(module, x, cu_seqlens=boundaries, use_reentrant=False)[0]
                else:
                    actual = module(x, cu_seqlens=boundaries)[0]
                ref_x = full_x.clone().requires_grad_()
                expected = kda_oracle(reference, ref_x, boundaries)
                local_expected = expected[:, start:start + 6 // sp_size]
                torch.testing.assert_close(actual, local_expected, rtol=5e-5, atol=5e-6)
                assert module.o_norm.weight is norm_parameter
                assert module.o_norm.forward.__func__ is norm_forward
                ((actual * target[:, start:start + 6 // sp_size]).sum() * sp_size).backward()
                (expected * target).sum().backward()
                expected_input_grad = ref_x.grad[:, start:start + 6 // sp_size] * sp_size
                torch.testing.assert_close(x.grad, expected_input_grad, rtol=5e-5, atol=5e-6)
                errors = dict(output=float((actual - local_expected).abs().max().detach()),
                              input_gradient=float((x.grad - expected_input_grad).abs().max()))
                for name, p in module.named_parameters():
                    rp = reference.get_parameter(name)
                    assert rp.grad.norm() > 0, name
                    if sp_size == 2:
                        assert isinstance(p.grad, DTensor), name
                        assert p.grad.placements == p.placements, name
                    elif name == "o_norm.weight":
                        assert p.grad.placements == (Partial(),)
                    gradient = full(p.grad)
                    torch.testing.assert_close(gradient, rp.grad, rtol=5e-5, atol=5e-6, msg=name)
                    factor = (gradient.flatten() @ rp.grad.flatten()) / rp.grad.square().sum()
                    assert .995 < factor < 1.005, (name, factor)
                    errors[name] = dict(max_error=float((gradient - rp.grad).abs().max()),
                                       reference_norm=float(rp.grad.norm()), factor=float(factor))
                    if sp_size == 1 and name == "o_norm.weight":
                        p.grad = p.grad.redistribute(placements=p.placements)
                optimizer.step()
                reference_optimizer.step()
                for name, p in module.named_parameters():
                    rp = reference.get_parameter(name)
                    torch.testing.assert_close(full(p), rp, rtol=5e-5, atol=5e-6, msg=name)
                    for state in ("exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(full(optimizer.state[p][state]),
                                                   reference_optimizer.state[rp][state], rtol=5e-5, atol=5e-6, msg=name)
                assert module.o_norm.weight is norm_parameter
                evidence["steps"].append(errors)
            handle.remove()
            assert len(identities) >= (4 if ac else 2)
            assert all(parameter is identities[0] for parameter in identities)
            evidence["materialized_norm_identity_count"] = len(identities)
            evidence["complete"] = True
    except BaseException as error:
        evidence["failure"] = repr(error)
        raise
    finally:
        (directory / f"rank-{rank}.json").write_text(json.dumps(evidence, indent=2))
        dist.destroy_process_group()


@pytest.mark.parametrize("family,direct,safe_gate,lower_bound,sp_size,ac", [
    ("hybrid", False, False, None, 1, False),
    ("vl", True, True, -2, 1, False),
    ("hybrid", True, False, -2, 2, False),
    ("vl", False, False, None, 2, True),
], ids=["hybrid-lora-softplus", "vl-direct-safe", "hybrid-fsdp-sp-bounded", "vl-fsdp-sp-ac"])
def test_native_kda_gradients(tmp_path, family, direct, safe_gate, lower_bound, sp_size, ac):
    mp.spawn(_worker, args=(str(tmp_path), family, direct, safe_gate, lower_bound, sp_size, ac),
             nprocs=4 if sp_size == 2 else 2, join=True)


def leaf_inputs(width=4):
    query = torch.linspace(-1e-4, 2e-4, 3 * width).reshape(1, 3, 1, width)
    return dict(q=query, k=query.flip(-1), v=torch.linspace(-.4, .8, 3 * width).reshape_as(query),
                g=torch.linspace(-.7, .5, 3 * width).reshape_as(query),
                beta=torch.tensor([[[.2], [.6], [.8]]]), A_log=torch.tensor([-.4]),
                dt_bias=torch.linspace(-.3, .3, width), cu_seqlens=torch.tensor([0, 2, 3]),
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True)


@pytest.mark.parametrize("safe,bound,width", [(True, None, 4), (True, -5.1, 4),
                                             (True, 0, 4), (True, .1, 4), (False, None, 257)])
def test_cpu_kda_native_contract_rejects_invalid_bound_or_width(safe, bound, width):
    with pytest.raises(ValueError, match="lower_bound|256"):
        cpu_chunk_kda(**leaf_inputs(width), safe_gate=safe, lower_bound=bound)


@pytest.mark.parametrize("field", ["batch", "key", "value_heads", "gate", "beta"])
def test_cpu_kda_native_contract_rejects_shape_leakage(field):
    inputs = leaf_inputs()
    if field == "batch":
        for name in ("q", "k", "v", "g", "beta"):
            inputs[name] = inputs[name].repeat(2, *([1] * (inputs[name].ndim - 1)))
    else:
        name = dict(key="k", value_heads="v", gate="g", beta="beta")[field]
        if field == "beta":
            inputs[name] = inputs[name][:, :-1]
        elif field == "value_heads":
            inputs[name] = inputs[name].repeat(1, 1, 2, 1)
        else:
            inputs[name] = inputs[name][..., :1]
    with pytest.raises(ValueError, match="shape|heads|batch"):
        cpu_chunk_kda(**inputs)


def test_cpu_kda_accepts_safe_bound_minus_five_and_returns_segment_states():
    inputs = leaf_inputs()
    output, states = cpu_chunk_kda(**inputs, safe_gate=True, lower_bound=-5, output_final_state=True)
    assert output.shape == inputs["v"].shape
    assert states.shape == (2, 1, 4, 4)
    assert states.dtype == torch.float32
    assert states.norm() > 0
    # Single-token segment starts with zero state; its final state is explicit.
    k = inputs["k"][:, 2].float()
    k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    expected_last = k[..., None] * (inputs["beta"][:, 2, ..., None] * inputs["v"][:, 2])[..., None, :]
    torch.testing.assert_close(states[-1:], expected_last)


def test_cpu_kda_small_norm_outputs_and_gradients_require_additive_epsilon():
    inputs = leaf_inputs()
    q = inputs["q"].clone().requires_grad_()
    k = inputs["k"].clone().requires_grad_()
    inputs.update(q=q, k=k)
    output, _ = cpu_chunk_kda(**inputs)
    target = torch.linspace(-.5, .9, output.numel()).reshape_as(output)
    actual_grads = torch.autograd.grad((output * target).sum(), (q, k))

    def literal(raw_q, raw_k, additive):
        if additive:
            nq = raw_q / (raw_q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            nk = raw_k / (raw_k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        else:
            nq = torch.nn.functional.normalize(raw_q, dim=-1)
            nk = torch.nn.functional.normalize(raw_k, dim=-1)
        decay = -inputs["A_log"].exp()[:, None] * torch.nn.functional.softplus(
            inputs["g"] + inputs["dt_bias"].reshape(1, 4))
        tokens = []
        for start, end in ((0, 2), (2, 3)):
            state = torch.zeros(1, 1, 4, 4)
            for token in range(start, end):
                state = state * decay[:, token].exp().unsqueeze(-1)
                residual = inputs["v"][:, token] - (nk[:, token].unsqueeze(-1) * state).sum(-2)
                state = state + nk[:, token].unsqueeze(-1) * (inputs["beta"][:, token].unsqueeze(-1) * residual).unsqueeze(-2)
                tokens.append((nq[:, token].unsqueeze(-1) * state).sum(-2) * .5)
        return torch.stack(tokens, 1)
    expected = literal(q, k, True)
    expected_grads = torch.autograd.grad((expected * target).sum(), (q, k))
    wrong = literal(q, k, False)
    wrong_grads = torch.autograd.grad((wrong * target).sum(), (q, k))
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-7)
    assert (output - wrong).abs().max() > .01
    for actual, expected, wrong in zip(actual_grads, expected_grads, wrong_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        assert (actual - wrong).norm() > actual.norm() * .5


def test_hybrid_safe_false_bound_uses_bounded_gate_and_vl_clears_bound():
    with cpu_bailing_leaves():
        hybrid = make_bailing_model("hybrid", no_kda_lora=True, kda_safe_gate=False,
                                    kda_lower_bound=-2, use_cache=False)
        vl = make_bailing_model("vl", kda_safe_gate=False, kda_lower_bound=-2, use_cache=False)
        module = hybrid.model.layers[0].attention
        assert module.safe_gate is False and module.lower_bound == -2
        assert vl.model.config.kda_lower_bound is None
        assert vl.model.layers[0].attention.lower_bound is None
        # Post-convolution Q/K have small magnitudes where additive epsilon matters.
        with torch.no_grad():
            module.q_proj.weight.mul_(.01)
            module.k_proj.weight.mul_(.01)
        reference = copy.deepcopy(module)
        softplus_reference = copy.deepcopy(reference)
        softplus_reference.lower_bound = None
        hybrid.on_parallelized({"sp": dict(sp_group=None, sp_size=1, sp_rank=0)})
        x = torch.sin(torch.arange(96).reshape(1, 6, 16).float() * .17).requires_grad_()
        ref_x = x.detach().clone().requires_grad_()
        boundaries = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
        output = module(x, cu_seqlens=boundaries)[0]
        expected = kda_oracle(reference, ref_x, boundaries)
        target = torch.linspace(-.7, .9, 96).reshape_as(x)
        (output * target).sum().backward()
        (expected * target).sum().backward()
        torch.testing.assert_close(output, expected, rtol=5e-5, atol=5e-6)
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=5e-5, atol=5e-6)
        for name, parameter in module.named_parameters():
            torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad,
                                       rtol=5e-5, atol=5e-6, msg=name)
        wrong = kda_oracle(softplus_reference, x.detach(), boundaries)
        separation = float((output - wrong).abs().max().detach())
        print(f"bounded_softplus_output_separation={separation:.9g}")
        assert separation > 1e-5


@pytest.mark.parametrize("offsets", [[1, 3], [0, 2], [0, 3, 2, 3], [0, 2, 2, 3]])
def test_cpu_kda_rejects_incomplete_or_unordered_packed_boundaries(offsets):
    inputs = leaf_inputs()
    inputs["cu_seqlens"] = torch.tensor(offsets, dtype=torch.int32)
    with pytest.raises(ValueError, match="boundaries"):
        cpu_chunk_kda(**inputs)


def test_cpu_kda_two_token_bounded_state_uses_exponential_rate():
    inputs = leaf_inputs()
    inputs["A_log"] = torch.tensor([torch.log(torch.tensor(2.))])
    _, states = cpu_chunk_kda(**inputs, safe_gate=False, lower_bound=-6, output_final_state=True)
    key = inputs["k"][:, :2]
    key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    first = key[:, 0, ..., None] * (inputs["beta"][:, 0, ..., None] * inputs["v"][:, 0])[..., None, :]
    raw = inputs["g"][:, 1] + inputs["dt_bias"].reshape(1, 4)
    # exp(A_log)=2 changes the supplied-bound decay of the nonzero first state.
    decay = torch.exp(-6 / (1 + torch.exp(-2 * raw)))
    decayed = first * decay.unsqueeze(-1)
    residual = inputs["v"][:, 1] - (key[:, 1].unsqueeze(-1) * decayed).sum(-2)
    expected = decayed + key[:, 1].unsqueeze(-1) * (inputs["beta"][:, 1].unsqueeze(-1) * residual).unsqueeze(-2)
    torch.testing.assert_close(states[:1], expected, rtol=1e-5, atol=1e-8)
    wrong_decay = torch.exp(-6 / (1 + torch.exp(-raw)))
    assert (first * (decay - wrong_decay).unsqueeze(-1)).norm() > 1e-6


def test_cpu_sigmoid_norm_closed_form_output_and_gradients():
    from .tp_helpers import cpu_sigmoid_norm
    x = torch.tensor([[1., 1.]], requires_grad=True)
    weight = torch.tensor([2., 3.], requires_grad=True)
    gate = torch.tensor([[0., torch.log(torch.tensor(3.))]], requires_grad=True)
    output = cpu_sigmoid_norm(x, gate, weight, None, activation="sigmoid", eps=0.)
    torch.testing.assert_close(output, torch.tensor([[1., 2.25]]))
    gradients = torch.autograd.grad(output.sum(), (x, weight, gate))
    torch.testing.assert_close(gradients[0], torch.tensor([[-.625, .625]]))
    torch.testing.assert_close(gradients[1], torch.tensor([.5, .75]))
    torch.testing.assert_close(gradients[2], torch.tensor([[.5, .5625]]))
