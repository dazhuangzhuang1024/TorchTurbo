"""Registered Bailing engine cases with an independent prepared-semantics oracle."""

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from importlib import import_module
from pathlib import Path
from typing import Any
from collections.abc import Iterator
from types import MethodType
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.distributed.tensor import DTensor, Partial
from torch.nn import functional as F
from transformers import PretrainedConfig

from torchturbo.engine import TorchTurboEngine
from torchturbo.engine.config import EngineConfig
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_layer import MoELayer
from torchturbo.parallel import FSDPPlanSpec
from tests.engine.backends.fsdp2.etp_checkpoint_helpers import read_hf
from tests.models.qwen3_moe.etp_helpers import full, literal_targets
from .tp_helpers import cpu_bailing_leaves, kda_oracle, make_bailing_config


@dataclass(frozen=True)
class BailingCase:
    family: str
    world: int
    sp: int
    ep: int
    aux: float = 0.0
    z: float = 0.0
    checkpoint: bool = False
    chunk: bool = False
    image: bool = False
    ac: str = "off"
    q_lora_rank: int | None = None
    gated_attention_proj_granularity_type: str | None = None
    no_kda_lora: bool = False
    kda_safe_gate: bool = False
    kda_lower_bound: float | None = None

    @property
    def data(self) -> int:
        return self.world // (2 * self.sp)

    @property
    def dense(self) -> int:
        return self.world // 2


LANGUAGE_CASES = (
    BailingCase("hybrid", 4, 1, 2, aux=.02, z=.01, q_lora_rank=8,
                gated_attention_proj_granularity_type="head_wise"),
    BailingCase("vl", 4, 2, 1, aux=.02, z=.01, chunk=True, ac="nonreentrant",
                gated_attention_proj_granularity_type="element_wise", no_kda_lora=True,
                kda_safe_gate=True, kda_lower_bound=-2),
    BailingCase("hybrid", 8, 2, 2, ac="nonreentrant"),
)


def configuration(case: BailingCase) -> PretrainedConfig:
    return make_bailing_config(case.family, q_lora_rank=case.q_lora_rank,
        gated_attention_proj_granularity_type=case.gated_attention_proj_granularity_type,
        no_kda_lora=case.no_kda_lora, kda_safe_gate=case.kda_safe_gate,
        kda_lower_bound=case.kda_lower_bound)


def raw_class(case: BailingCase) -> type[nn.Module]:
    path = ("bailing_hybrid.modeling_bailing_moe_v3" if case.family == "hybrid"
            else "bailing_moe_v3_vl.modeling_bailing_moe_v3_vl")
    modeling = import_module("torchturbo.models." + path)
    return (modeling.BailingMoeV3ForCausalLM if case.family == "hybrid"
            else modeling.BailingMoeV3VLForConditionalGeneration)


def ordinary_model(case: BailingCase) -> nn.Module:
    config = configuration(case)
    config._attn_implementation = "eager"
    if case.family == "vl":
        config.text_config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"
    return raw_class(case)(config)


def write_bailing_source(directory: Path, case: BailingCase) -> None:
    directory.mkdir(parents=True)
    with cpu_bailing_leaves():
        torch.manual_seed(113)
        model = ordinary_model(case)
        assert model.model.config.use_cache is True
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                numbers = torch.arange(parameter.numel()).reshape(parameter.shape).float()
                if name.endswith("dt_bias"):
                    parameter.copy_(torch.linspace(-.3, .3, parameter.numel()))
                elif name.endswith("A_log"):
                    parameter.copy_(torch.linspace(-1.2, -.4, parameter.numel()))
                elif "norm" in name and parameter.ndim == 1:
                    parameter.copy_(torch.linspace(.7, 1.3, parameter.numel()))
                else:
                    phase = .31
                    if ".mlp.experts." in name:
                        expert_index = int(name.split(".experts.")[1].split(".")[0])
                        phase += expert_index * .27
                    parameter.copy_(.12 * torch.sin(numbers * .17 + phase))
            model.model.word_embeddings.weight[:16, 0] = .8
            model.model.word_embeddings.weight[16:, 0] = -.8
            gate = model.model.layers[1].mlp.gate
            gate.weight.zero_()
            gate.weight[:, 0] = torch.tensor([.6, .8, 1., 1.2, -.6, -.8, -1., -1.2])
            gate.expert_bias.copy_(torch.linspace(-.035, .035, 8))
        values = {name: value.clone().contiguous() for name, value in model.state_dict().items()}
        save_file(values, str(directory / "model.safetensors"))
        (directory / "model.safetensors.index.json").write_text(json.dumps(dict(
            metadata=dict(total_size=sum(v.numel() * v.element_size() for v in values.values())),
            weight_map=dict.fromkeys(values, "model.safetensors"))))
        model.config.save_pretrained(directory)


def prepare_bailing(source: Path, case: BailingCase) -> tuple[
        TorchTurboEngine, nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    config = EngineConfig.from_dict(dict(mixed_precision="fp32", sp_size=case.sp,
        tp_size=2, ep_size=case.ep, etp_size=2))
    config.mfu.enabled = False
    config.loss_chunk.enabled, config.loss_chunk.chunk_size = case.chunk, 2
    config.moe.aux_loss_coeff, config.moe.z_loss_coeff = case.aux, case.z
    config.activation_checkpointing.enabled = case.ac != "off"
    config.activation_checkpointing.use_reentrant = False
    config.parallel_plan.plan["model.layers.*"] = FSDPPlanSpec(activation_checkpoint=case.ac != "off")
    config.checkpoint.hf_export = "final"
    engine = TorchTurboEngine(config)
    plan = import_module("torchturbo.models." + ("bailing_hybrid" if case.family == "hybrid"
                                                else "bailing_moe_v3_vl") + ".parallel_plan")
    name = "_replace_moe_only" if case.family == "hybrid" else "_replace_vl_moe"
    original = plan.__dict__[name]
    calls = []
    def replace(original_module, *args, **kwargs):
        assert kwargs["parallel_context"]["tp"]["tp_size"] == 2
        shared = original_module.shared_experts if isinstance(original_module, sparse_type(case)) else None
        result = original(original_module, *args, **kwargs)
        if shared is not None:
            assert result.shared_experts is shared
        calls.append(result)
        return result
    with patch.object(plan, name, replace):
        model = engine.prepare_model(str(source), torch_dtype=torch.float32)
    assert len(calls) == 2 and calls[0] is model.model.layers[0].mlp
    assert isinstance(model.model.layers[1].mlp, MoELayer)
    assert isinstance(model.model.layers[1].mlp.experts, GroupedMLP)
    if case.family == "hybrid":
        from torchturbo.models.bailing_hybrid.model import TurboBailingMoeV3ForCausalLM
        assert isinstance(model, TurboBailingMoeV3ForCausalLM)
    else:
        from torchturbo.models.bailing_moe_v3_vl.model import TurboBailingMoeV3VLForConditionalGeneration
        assert isinstance(model, TurboBailingMoeV3VLForConditionalGeneration)
    assert model.model.config.use_cache is False
    assert model.model.config.num_attention_heads == 4 and model.model.config.num_key_value_heads == 1
    assert model.model.layers[0].attention.num_heads == 2
    assert model.model.layers[1].attention.num_heads == 2
    assert all(not p.is_meta for p in list(model.parameters()) + list(model.buffers()))
    bias = read_hf(source)["model.layers.1.mlp.gate.expert_bias"]
    torch.testing.assert_close(full(model.model.layers[1].mlp.router.expert_bias), bias, rtol=0, atol=0)
    experts = model.model.layers[1].mlp.experts
    assert tuple(experts.weight1.shape) == (8 // case.ep * 16, 2, 12)
    assert tuple(experts.weight2.shape) == (8 // case.ep, 12, 16)
    assert experts.weight1.to_local().shape[-1] == 6
    assert experts.weight2.to_local().shape[-2] == 6

    attention = model.model.layers[1].attention
    query = attention.q_proj if case.q_lora_rank is None else attention.q_b_proj
    dense_rank = engine.backend._parallel_context.dense_fsdp_mesh.get_local_rank()
    for parameter, tp_rows in ((model.model.layers[0].mlp.gate_proj.weight, 12),
                               (model.model.layers[1].mlp.shared_experts.gate_proj.weight, 6),
                               (query.weight, 24), (attention.kv_b_proj.weight, 16)):
        rows_per_shard = math.ceil(tp_rows / case.dense)
        local_rows = max(0, min(rows_per_shard, tp_rows - dense_rank * rows_per_shard))
        assert parameter.to_local().shape[-2] == local_rows
    optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=.003, betas=(.8, .95),
        weight_decay=.03, eps=1e-6, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
    return engine, model, optimizer, scheduler


def sparse_type(case: BailingCase) -> type[nn.Module]:
    return import_module(raw_class(case).__module__).BailingMoeV3SparseMoeBlock


def bailing_batch(data: int, micro: int, step: int, case: BailingCase) -> dict:
    if case.image:
        ids = torch.tensor([[2, 62, 60, 63, 3, 4]])
        cu = torch.tensor([0, 6], dtype=torch.int32)
        return dict(input_ids=ids, labels=ids.clone(), pixel_values=torch.linspace(-.8, .9, 96).reshape(4, 24),
            image_grid_thw=torch.tensor([[1, 2, 2]]), mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0]]),
            cu_seq_lens_q=cu, cu_seq_lens_k=cu.clone(), max_length_q=6, max_length_k=6)
    bounds = (0, 1, 3, 6) if data % 2 == 0 else (0, 2, 6, 8)
    length = bounds[-1] + 1
    ids = torch.arange(length).unsqueeze(0) % 8 + (1 if micro == 0 and step == 0 else 20)
    labels = ids.clone()
    if step == 0 and micro == 1:
        labels.fill_(-100)
    elif data % 2:
        labels[0, 4] = -100
    positions = [i-start for start, end in zip(bounds, bounds[1:]) for i in range(start, end)] + [0]
    cu = torch.tensor(bounds, dtype=torch.int32)
    return dict(input_ids=ids, labels=labels, position_ids=torch.tensor([positions]),
        cu_seq_lens_q=cu, cu_seq_lens_k=cu.clone(), max_length_q=max(b-a for a,b in zip(bounds,bounds[1:])),
        max_length_k=max(b-a for a,b in zip(bounds,bounds[1:])))


def literal_routing(logits: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Selected FP32 sigmoid/top2-group/nonzero-scale matrix only."""
    scores = logits.float().sigmoid()
    selection = scores + bias
    groups = selection.reshape(-1, 2, 4).topk(2, -1).values.sum(-1)
    chosen = groups.topk(1, -1).indices
    group_mask = torch.zeros_like(groups, dtype=torch.bool).scatter_(1, chosen, True)
    mask = group_mask.repeat_interleave(4, dim=-1)
    ids = selection.masked_fill(~mask, float("-inf")).topk(2, -1).indices
    selected = scores.gather(-1, ids)
    probs = selected / (selected.sum(-1, keepdim=True) + 1e-20) * 1.3
    return ids, torch.zeros_like(scores).scatter(1, ids, probs)


def expert_function(expert: nn.Module, hidden: torch.Tensor, limit: float) -> torch.Tensor:
    gate = F.silu(F.linear(hidden, expert.gate_proj.weight)).clamp(max=limit)
    up = F.linear(hidden, expert.up_proj.weight).clamp(-limit, limit)
    return F.linear(gate * up, expert.down_proj.weight)


def independent_moe(module: nn.Module, hidden: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    flat = hidden.reshape(-1, hidden.size(-1))
    module.literal_logits = F.linear(flat.float(), module.gate.weight.float())
    module.literal_ids, module.literal_probs = literal_routing(module.literal_logits, module.gate.expert_bias)
    module.literal_probs.retain_grad()
    routed = torch.zeros_like(flat)
    for index, expert in enumerate(module.experts):
        routed = routed + expert_function(expert, flat, .4) * module.literal_probs[:, index:index+1]
    result = routed.reshape_as(hidden) + module.shared_experts(hidden)
    return result, (module.literal_logits.reshape(*hidden.shape[:2], 8), module.literal_ids.reshape(*hidden.shape[:2], 2))


def independent_kda(module: nn.Module, hidden_states: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, None, None]:
    return kda_oracle(module, hidden_states, module.literal_boundaries), None, None


def segment_mla(module: nn.Module, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor | None, scaling: float, **kwargs: Any) -> tuple[torch.Tensor, None]:
    assert query.size(1) == key.size(1) == value.size(1) == 4
    bounds = module.literal_boundaries.tolist()
    pieces = [F.scaled_dot_product_attention(query[:, :, start:end], key[:, :, start:end],
        value[:, :, start:end], is_causal=True, dropout_p=0., scale=scaling)
        for start, end in zip(bounds, bounds[1:])]
    output = F.pad(torch.cat(pieces, 2), (0, 0, 0, query.size(2) - bounds[-1]))
    return output.transpose(1, 2).contiguous(), None


@contextmanager
def reference_mla_scope(case: BailingCase) -> Iterator[None]:
    modeling = import_module(raw_class(case).__module__)
    original = modeling.eager_attention_forward
    def leaf(module, *args, **kwargs):
        if isinstance(module, modeling.BailingMoeV3MultiLatentAttention):
            return segment_mla(module, *args, **kwargs)
        return original(module, *args, **kwargs)
    with patch.object(modeling, "eager_attention_forward", leaf):
        yield


def reference_model(source: Path, case: BailingCase) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    model = ordinary_model(case)
    model.load_state_dict(read_hf(source))
    moe = model.model.layers[1].mlp
    moe.forward = MethodType(independent_moe, moe)
    kda = model.model.layers[0].attention
    kda.forward = MethodType(independent_kda, kda)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if name.endswith(".bias") or p.ndim == 1 else decay).append(p)
    optimizer = torch.optim.AdamW([dict(params=decay), dict(params=no_decay, weight_decay=0.)],
        lr=.003, betas=(.8, .95), weight_decay=.03, eps=1e-6, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
    return model, optimizer, scheduler


class BailingEvidence:
    def __init__(self) -> None:
        self.errors, self.scales, self.routes, self.steps = {}, {}, [], []
        self.reference_gradient_norms = {}
        self.moments = {}

    def close(self, actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
        actual, expected = full(actual), full(expected)
        error = float((actual - expected).abs().max()) if actual.numel() else 0.
        self.errors[name] = max(error, self.errors.get(name, 0.))
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5, msg=lambda message: f"{name}: max_error={error}\n{message}")

    def gradient(self, actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
        self.close(actual, expected, "gradient " + name)
        self.reference_gradient_norms.setdefault(name, []).append(float(expected.norm()))
        if expected.square().sum() > 0:
            scale = float((full(actual).double() * expected.double()).sum() / expected.double().square().sum())
            self.scales.setdefault(name, []).append(scale)
            assert .995 < scale < 1.005, (name, scale)


def observe(model: nn.Module) -> tuple[dict, dict, list, list]:
    stages, inputs, routes, handles = {}, {0: [], 1: []}, [], []
    for index, layer in enumerate(model.model.layers):
        def retain(module, args, index=index):
            if torch.is_grad_enabled():
                args[0].retain_grad()
                inputs[index].append(args[0])
        handles.append(layer.register_forward_pre_hook(retain))
        names = ["input_layernorm", "attention", "post_attention_layernorm", "mlp"]
        if index == 1:
            names.append("mlp.shared_experts")
        for name in names:
            def stage(module, args, output, key=f"{index}.{name}"):
                stages[key] = (output[0] if isinstance(output, tuple) else output).detach().clone()
            handles.append(layer.get_submodule(name).register_forward_hook(stage))
    handles.append(model.model.norm.register_forward_hook(lambda module, args, output: stages.__setitem__("final", output.detach().clone())))
    if isinstance(model.model.layers[1].mlp, MoELayer):
        def route(module, args, output):
            probs, routing, logits = output
            if torch.is_grad_enabled():
                probs.retain_grad()
                routes.append((probs, routing.detach(), logits.detach()))
        handles.append(model.model.layers[1].mlp.router.register_forward_hook(route))
    return stages, inputs, routes, handles


def auxiliary(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = logits.size(0)
    if length == 0:
        zero = logits.sum() * 0
        return zero, zero, logits.new_zeros(8)
    scores = logits.float().sigmoid()
    scores = scores / (scores.sum(-1, keepdim=True) + 1e-20)
    counts = F.one_hot(scores.topk(2, -1).indices, 8).sum((0, 1)).float()
    aux = (scores.sum(0) * counts).sum() * 8 / (2 * length ** 2)
    z = logits.logsumexp(-1).square().mean()
    return aux, z, counts


@contextmanager
def image_observer(model: nn.Module, case: BailingCase, trace: dict) -> Iterator[None]:
    """Observe real global fusion before SP and the generated local M-RoPE rows."""
    if not case.image:
        yield
        return
    original = model.model._forward_text
    def text_entry(**kwargs):
        trace["global_fused"] = kwargs["inputs_embeds"].detach().clone()
        trace["global_positions"] = kwargs["position_ids"].detach().clone()
        return original(**kwargs)
    handles = []
    def record(key):
        def hook(module, args, output):
            trace[key] = output.detach().clone()
        return hook
    handles.append(model.model.word_embeddings.register_forward_hook(record("word_embeddings")))
    handles.append(model.linear_proj.register_forward_hook(record("projected_image")))
    def visual_input(module, args, kwargs):
        trace["pixels"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
    handles.append(model.model.visual.register_forward_pre_hook(visual_input, with_kwargs=True))
    def positions(module, args, kwargs):
        trace["local_positions"] = (args[1] if len(args) > 1 else kwargs["position_ids"]).detach().clone()
    handles.append(model.model.rotary_emb.register_forward_pre_hook(positions, with_kwargs=True))
    try:
        with patch.object(model.model, "_forward_text", text_entry):
            yield
    finally:
        for handle in handles:
            handle.remove()


def oracle_step(model: nn.Module, case: BailingCase, step: int, micros: int) -> tuple[int, torch.Tensor, dict]:
    batches = {(data, micro): bailing_batch(data, micro, step, case)
               for data in range(case.data) for micro in range(micros)}
    count = sum(int((literal_targets(raw, raw["input_ids"].size(1)) != -100).sum()) for raw in batches.values())
    records, report = {}, torch.tensor(0.)
    for (data, micro), raw in batches.items():
        extent = raw["input_ids"].size(1)
        extent += (-extent) % case.sp
        tail = extent - raw["input_ids"].size(1)
        valid_bounds = raw["cu_seq_lens_q"]
        full_bounds = valid_bounds if int(valid_bounds[-1]) == extent else torch.cat((valid_bounds, torch.tensor([extent], dtype=torch.int32)))
        model.model.layers[0].attention.literal_boundaries = full_bounds
        model.model.layers[1].attention.literal_boundaries = valid_bounds
        stages, observed, _, handles = observe(model)
        inputs = {name: value for name, value in raw.items() if name != "labels"}
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, tail))
        if "position_ids" in inputs:
            inputs["position_ids"] = F.pad(inputs["position_ids"], (0, tail))
        image_trace = {}
        with reference_mla_scope(case), image_observer(model, case, image_trace):
            output = model(**inputs, use_cache=False).logits
        main = F.cross_entropy(output.reshape(-1, 64), literal_targets(raw, extent).reshape(-1), ignore_index=-100, reduction="sum") / count
        moe = model.model.layers[1].mlp
        length = int(valid_bounds[-1])
        valid_logits = moe.literal_logits[:length]
        aux, z, counts = auxiliary(valid_logits)
        objective = main + (case.aux * aux + case.z * z) / (case.data * micros)
        metric_aux, metric_z = [], []
        scores = valid_logits.sigmoid()
        scores = scores / (scores.sum(-1, keepdim=True) + 1e-20)
        local_length = extent // case.sp
        for sp in range(case.sp):
            start, end = sp * local_length, min((sp + 1) * local_length, length)
            metric_aux.append((scores[start:end].sum(0) * counts).sum() * 8 * case.sp / (2 * length ** 2))
            metric_z.append(valid_logits[start:end].logsumexp(-1).square().sum() * case.sp / length)
        objective.backward()
        records[data, micro] = dict(logits=output.detach(), stages=stages,
            input=[observed[i][0].grad.detach() for i in range(2)], probs=moe.literal_probs.detach(),
            ids=moe.literal_ids.detach(), probability_grad=moe.literal_probs.grad.detach(),
            metric_aux=metric_aux, metric_z=metric_z, main=main.detach(), bounds=full_bounds.tolist(), image_trace=image_trace)
        report += main.detach()
        for handle in handles:
            handle.remove()
    return count, report, records


def converted_fields(converter, name: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
    value = full(value)
    if name.endswith(".experts.weight1"):
        return converter.get_local_ep_w1_hf_state_dict(name, value)
    if name.endswith(".experts.weight2"):
        return converter.get_local_ep_w2_hf_state_dict(name, value)
    return {converter.to_hf_name(name): value}


def compare_field(evidence: BailingEvidence, converter, reference: nn.Module, name: str,
                  value: torch.Tensor, field: str, ref_optimizer: torch.optim.Optimizer | None = None) -> None:
    for source_name, actual in converted_fields(converter, name, value).items():
        parameter = reference.get_parameter(source_name)
        expected = parameter if field in ("parameter", "initial") else parameter.grad
        if ref_optimizer is not None:
            expected = ref_optimizer.state[parameter][field]
            actual_norm = float(full(actual).double().norm())
            expected_norm = float(expected.double().norm())
            error_norm = float((full(actual).double() - expected.double()).norm())
            if expected_norm == 0.:
                assert torch.count_nonzero(full(actual)) == 0, source_name
                relative_error = 0.
            else:
                relative_error = error_norm / expected_norm
                assert relative_error <= 1e-3, (source_name, field, relative_error)
            evidence.moments.setdefault(field + " " + source_name, []).append(dict(
                step=int(ref_optimizer.state[parameter]["step"]),
                actual_norm=actual_norm, expected_norm=expected_norm,
                error_norm=error_norm, relative_error=relative_error))
        if field == "gradient":
            evidence.gradient(actual, expected, source_name)
        else:
            evidence.close(actual, expected, field + " " + source_name)


def train_step(engine, model, optimizer, scheduler, reference, ref_optimizer, ref_scheduler,
               case: BailingCase, step: int, micros: int, evidence: BailingEvidence) -> float:
    optimizer.zero_grad(set_to_none=True)
    ref_optimizer.zero_grad(set_to_none=True)
    count, expected_report, records = oracle_step(reference, case, step, micros)
    context = engine.backend._parallel_context
    data, sp = context.data_dp_mesh.get_local_rank(), context.sp_mesh.get_local_rank()
    batches = [bailing_batch(data, micro, step, case) for micro in range(micros)]
    prepared, actual_count = engine.prepare_loss_step(model, batches, torch.device("cpu"))
    assert actual_count == count
    converter = model.get_weight_converter(engine.backend.get_parallel_context(), str(engine.model_name_or_path))
    router = model.model.layers[1].mlp.router
    statistics_before = full(router.local_tokens_per_expert) if "local_tokens_per_expert" in router._buffers else torch.zeros(8)
    metric_before = {key: value.clone() for key, value in engine.metric_tracker.totals.items()}
    metric_counts = dict(engine.metric_tracker.counts)
    report, expected_statistics = torch.tensor(0.), torch.zeros(8)
    for micro, inputs in enumerate(prepared):
        stages, observed, routes, handles = observe(model)
        image_trace = {}
        with image_observer(model, case, image_trace):
            loss, output = engine.compute_prepared_loss(model, inputs, True, num_items_in_batch=count,
                                                       gradient_accumulation_steps=micros)
        assert not output.main_loss.requires_grad
        report += output.main_loss
        logits = output.logits if output.logits is not None else F.linear(stages["final"], full(model.lm_head.weight))
        expected = records[data, micro]
        local_length = logits.size(1)
        begin, end = sp * local_length, min((sp + 1) * local_length, int(inputs["cu_seq_lens_q"][-1]))
        for name, actual in stages.items():
            wanted = expected["stages"][name]
            actual, wanted = actual.reshape(1, -1, actual.size(-1)), wanted.reshape(1, -1, wanted.size(-1))
            evidence.close(actual, wanted[:, begin:begin+local_length], "stage " + name)
        evidence.close(logits, expected["logits"][:, begin:begin+local_length], "logits")
        if case.image:
            wanted_trace = expected["image_trace"]
            assert image_trace["global_positions"].shape == (3, 1, 6)
            assert image_trace["local_positions"].shape == (3, 1, 3)
            assert image_trace["global_fused"].shape == (1, 6, 16)
            evidence.close(image_trace["global_positions"], wanted_trace["global_positions"], "M-RoPE global")
            evidence.close(image_trace["local_positions"], wanted_trace["local_positions"][:, :, begin:begin+local_length], "M-RoPE local")
            evidence.close(image_trace["global_fused"], wanted_trace["global_fused"], "global image fusion")
            evidence.close(stages["0.input_layernorm"], expected["stages"]["0.input_layernorm"][:, begin:begin+local_length], "post-fusion text SP")
            assert image_trace["pixels"].shape == (4, 24)
            evidence.close(image_trace["global_fused"][:, 2], image_trace["projected_image"], "projected placeholder")
            assert not torch.equal(image_trace["global_fused"][:, 2], image_trace["word_embeddings"][:, 2])
        loss.backward()
        for index in range(2):
            active = [value for value in observed[index] if value.grad is not None]
            assert len(active) == 1
            evidence.close(active[0].grad, expected["input"][index][:, begin:begin+local_length] * case.dense, f"input gradient {index}")
        active_routes = [entry for entry in routes if entry[0].grad is not None]
        assert len(active_routes) == 1
        probs, routing, _ = active_routes[0]
        ids = expected["ids"][begin:begin+local_length]
        route_map = F.one_hot(ids, 8).sum(1).bool()
        assert torch.equal(routing, route_map)
        evidence.close(probs, expected["probs"][begin:begin+local_length], "probabilities")
        evidence.close(probs.grad[route_map], (expected["probability_grad"][begin:begin+local_length] * case.dense)[route_map], "selected probability gradient")
        assert torch.count_nonzero(probs.grad[~route_map]) == 0
        valid_counts = route_map[:end-begin].sum(0)
        expected_statistics += valid_counts
        scores = torch.sigmoid(records[data, micro]["stages"]["1.post_attention_layernorm"].reshape(-1, 16) @ reference.model.layers[1].mlp.gate.weight.detach().t())
        selection = scores + reference.model.layers[1].mlp.gate.expert_bias.detach()
        group_scores = selection.reshape(-1, 2, 4).topk(2, -1).values.sum(-1)
        group_margin = float((group_scores.max(-1).values - group_scores.min(-1).values)[:int(inputs["cu_seq_lens_q"][-1])].min())
        selected_group = group_scores.argmax(-1)
        within_group = selection.reshape(-1, 2, 4)[torch.arange(selection.size(0)), selected_group]
        top_scores = within_group.topk(3, -1).values
        expert_margin = float((top_scores[:, 1] - top_scores[:, 2])[:int(inputs["cu_seq_lens_q"][-1])].min())
        assert expert_margin > .001
        record = dict(step=step, micro=micro, data=data, sp=sp, counts=valid_counts.tolist(), ids=ids.tolist(),
                      bounds=expected["bounds"], group_margin=group_margin, expert_margin=expert_margin,
                      local_valid_tokens=end-begin, full_replica_nonignored_unshifted_labels=int((inputs["labels"] != -100).sum()))
        assert group_margin > .01
        if case.image:
            record["image_trace"] = {name: value.tolist() for name, value in image_trace.items()}
        if case.ep > 1:
            dispatcher = model.model.layers[1].mlp.token_dispatcher
            counts = route_map.sum(0)
            input_splits = counts.reshape(case.ep, 8 // case.ep).sum(1).tolist()
            assert dispatcher.input_splits.tolist() == input_splits
            owner = context.ep_mesh.get_local_rank()
            domain = (data * case.sp + sp) // case.ep
            output_splits = []
            for peer in range(case.ep):
                peer_data, peer_sp = divmod(domain * case.ep + peer, case.sp)
                peer_ids = records[peer_data, micro]["ids"].chunk(case.sp)[peer_sp]
                peer_counts = F.one_hot(peer_ids, 8).sum((0, 1))
                output_splits.append(int(peer_counts.chunk(case.ep)[owner].sum()))
            assert dispatcher.output_splits.tolist() == output_splits
            record.update(owner=owner, input_splits=input_splits, output_splits=output_splits)
        evidence.routes.append(record)
        for metric, coefficient, field in (("aux_loss", case.aux, "metric_aux"), ("z_loss", case.z, "metric_z")):
            if coefficient:
                evidence.close(router._metrics[metric], expected[field][sp], metric)
                evidence.close(router._metrics[metric + "_scaled"], expected[field][sp] * coefficient, metric + " scaled")
        engine.collect_metrics(model)
        assert router.pop_metrics() == {}
        for handle in handles:
            handle.remove()
    evidence.close(report, expected_report, "main report")
    evidence.close(full(router.local_tokens_per_expert) - statistics_before, expected_statistics, "route statistics")
    for metric, coefficient, field in (("aux_loss", case.aux, "metric_aux"), ("z_loss", case.z, "metric_z")):
        if coefficient:
            expected = sum(records[data, micro][field][sp] for micro in range(micros))
            for suffix, scale in (("", 1.), ("_scaled", coefficient)):
                key = metric + suffix
                evidence.close(engine.metric_tracker.totals[key] - metric_before.get(key, 0.), expected * scale, key + " total")
                assert engine.metric_tracker.counts[key] == metric_counts.get(key, 0) + micros
    inactive = []
    before = {name: full(p) for name, p in model.named_parameters()}
    reference_before = {name: p.detach().clone() for name, p in reference.named_parameters()}
    for name, p in model.named_parameters():
        source_fields = converted_fields(converter, name, p)
        expected_grads = [reference.get_parameter(source_name).grad for source_name in source_fields]
        if all(value is None for value in expected_grads):
            assert p.grad is None or torch.count_nonzero(full(p.grad)) == 0, name
            inactive.append(dict(name=name, gradient="None" if p.grad is None else "zero"))
            continue
        assert p.grad is not None, name
        assert isinstance(p.grad, DTensor) and p.grad.placements == p.placements, name
        assert p.grad.to_local().shape == p.to_local().shape, name
        assert not any(isinstance(place, Partial) for place in p.grad.placements)
        compare_field(evidence, converter, reference, name, p.grad, "gradient")
    norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in reference.parameters() if p.grad is not None]))
    assert norm > .15
    evidence.close(engine.backend.clip_grad_norm_(model.parameters(), .15), norm, "norm")
    torch.nn.utils.clip_grad_norm_(reference.parameters(), .15)
    for name, p in model.named_parameters():
        if p.grad is not None and name not in {item["name"] for item in inactive}:
            compare_field(evidence, converter, reference, name, p.grad, "clipped")
    optimizer.step()
    ref_optimizer.step()
    scheduler.step()
    ref_scheduler.step()
    for name, p in model.named_parameters():
        if name in {item["name"] for item in inactive}:
            if p.grad is None:
                assert p not in optimizer.state
                torch.testing.assert_close(full(p), before[name], rtol=0, atol=0, msg=name)
            else:
                for field in ("exp_avg", "exp_avg_sq"):
                    assert torch.count_nonzero(full(optimizer.state[p][field])) == 0
                decay = 0. if name.endswith(".bias") or p.ndim == 1 else .03
                evidence.close(p, before[name] * (1 - .003 * .8 ** step * decay), "inactive decay " + name)
            continue
        compare_field(evidence, converter, reference, name, p, "parameter")
        for source_name, actual_update in converted_fields(converter, name, full(p) - before[name]).items():
            evidence.close(actual_update, reference.get_parameter(source_name).detach() - reference_before[source_name], "update " + source_name)
        for field in ("exp_avg", "exp_avg_sq"):
            compare_field(evidence, converter, reference, name, optimizer.state[p][field], field, ref_optimizer)
        assert float(optimizer.state[p]["step"]) == step + 1
    assert scheduler.last_epoch == ref_scheduler.last_epoch
    assert all(lr == pytest.approx(.003 * .8 ** (step + 1)) for lr in scheduler.get_last_lr())
    assert all(lr == pytest.approx(.003 * .8 ** (step + 1)) for lr in ref_scheduler.get_last_lr())
    evidence.steps.append(dict(step=step, micros=micros, global_count=count, norm=float(norm),
        reported_ce=float(report), expected_ce=float(expected_report),
        metrics={key: float(value - metric_before.get(key, 0.)) for key,value in engine.metric_tracker.totals.items()},
        metric_counts={key: value - metric_counts.get(key, 0) for key,value in engine.metric_tracker.counts.items()},
        inactive=[{**item, "slots": sorted(optimizer.state.get(model.get_parameter(item["name"]), {}))} for item in inactive],
        optimizer_slot_names=[name for name,p in model.named_parameters() if p in optimizer.state]))
    return float(norm)


def run_bailing_case(rank: int, directory: str, case: BailingCase) -> None:
    from dataclasses import asdict
    directory = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank,
                            world_size=case.world, timeout=timedelta(seconds=120))
    evidence, events = BailingEvidence(), []
    result = dict(rank=rank, case=asdict(case), complete=False)
    try:
        with cpu_bailing_leaves(events):
            engine, model, optimizer, scheduler = prepare_bailing(directory / "source", case)
            reference, ref_optimizer, ref_scheduler = reference_model(directory / "source", case)
            converter = model.get_weight_converter(engine.backend.get_parallel_context(), str(directory / "source"))
            for name, p in model.named_parameters():
                compare_field(evidence, converter, reference, name, p, "initial")
            assert all(p.grad is None for p in model.parameters())
            assert len(optimizer.state) == 0
            result["after_prepare_optimizer_slots"] = []
            result["after_prepare_inactive"] = [dict(name=name, gradient="None", slots=[])
                for name, p in model.named_parameters()
                if ".visual." in name or name.startswith("linear_proj.")]
            bias_before = full(model.model.layers[1].mlp.router.expert_bias)
            result["source_cache"] = reference.model.config.use_cache
            result["runtime_cache"] = model.model.config.use_cache
            for step, micros in enumerate((1,) if case.image else (2, 1)):
                train_step(engine, model, optimizer, scheduler, reference, ref_optimizer,
                           ref_scheduler, case, step, micros, evidence)
            torch.testing.assert_close(full(model.model.layers[1].mlp.router.expert_bias), bias_before, rtol=0, atol=0)
            if case.image:
                required = ["model.visual.patch_embed.proj.weight", "linear_proj.0.weight", "linear_proj.2.weight",
                            "model.layers.0.attention.q_proj.weight", "model.layers.1.attention.kv_b_proj.weight",
                            "model.layers.1.mlp.experts.weight1", "model.layers.1.mlp.shared_experts.gate_proj.weight"]
                result["image_gradients"] = {name: float(full(model.get_parameter(name).grad).norm()) for name in required}
                assert all(value > 0 for value in result["image_gradients"].values())
                assert any(event["op"] == "vision_segment_attention" for event in events)
                result["reference_vision_path"] = "original eager per-segment attention"
            else:
                assert any(event["boundaries"][-1] == event["extent"] and len(event["boundaries"]) == 5
                           for event in events if event["op"] == "causal_conv1d")
            for name, norms in evidence.reference_gradient_norms.items():
                if ".attention." in name or ".mlp.gate.weight" in name or ".mlp.shared_experts." in name:
                    assert max(norms) > 1e-10, name
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for owner in range(case.ep):
                    names = [f"model.layers.1.mlp.experts.{index}.{projection}.weight"
                             for index in range(owner * (8 // case.ep), (owner + 1) * (8 // case.ep))]
                    observed_norms = [max(evidence.reference_gradient_norms[name]) for name in names
                                      if name in evidence.reference_gradient_norms]
                    if observed_norms:
                        assert max(observed_norms) > 1e-10
            rotary_gradient = reference.model.layers[1].attention.kv_a_proj_with_mqa.weight.grad[8:]
            assert rotary_gradient.norm() > 1e-10
            result["rotary_key_gradient_norm"] = float(rotary_gradient.norm())
            result["layouts"] = {name: dict(shape=list(p.shape), local_shape=list(p.to_local().shape),
                placements=[str(place) for place in p.placements],
                gradient_placements=None if p.grad is None else [str(place) for place in p.grad.placements])
                for name, p in model.named_parameters()}
            result["complete"] = True
    except BaseException as error:
        result["failure"] = repr(error)
        raise
    finally:
        result.update(errors=evidence.errors, scales=evidence.scales, routes=evidence.routes,
                      moments=evidence.moments,
                      reference_gradient_norms=evidence.reference_gradient_norms,
                      steps=evidence.steps, kernel_events=events)
        (directory / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()
