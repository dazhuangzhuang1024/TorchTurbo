"""Real mixed-decoder engine acceptance with independent packed math."""

import copy
import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.distributed.tensor import Partial
from torch.nn import functional as F

from torchturbo.engine import TorchTurboEngine
from torchturbo.engine.config import EngineConfig
from torchturbo.engine.optimizers import OptimizerClassPlan
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_layer import MoELayer
from torchturbo.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling
from torchturbo.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from torchturbo.models.qwen3_5_moe.model import TurboQwen3_5MoeForConditionalGeneration
from torchturbo.models.qwen3_5_moe_text.model import TurboQwen3_5MoeTextForCausalLM
from torchturbo.parallel import FSDPPlanSpec
from tests.engine.backends.fsdp2.etp_checkpoint_helpers import read_hf
from tests.models.qwen3_moe.etp_helpers import batch, expected_local, full, literal_targets

from .tp_helpers import gdn_oracle, initialize_gdn, install_cpu_leaves, text_config


@dataclass(frozen=True)
class Case:
    world: int = 4
    sp: int = 1
    ep: int = 1
    tp: int = 2
    experts: int = 4
    aux: float = 0.
    z: float = 0.
    ac: str = "off"
    clip: bool = False
    chunk: bool = False
    bf16: bool = False
    empty_owner: bool = False
    key_repeat: bool = False
    wrapper: bool = False

    @property
    def data(self):
        return self.world // (self.tp * self.sp)

    @property
    def dense(self):
        return self.world // self.tp


def backbone(model, case):
    return model.model.language_model if case.wrapper else model.model


def configuration(case):
    config = text_config(num_experts=case.experts, output_router_logits=False,
                         router_aux_loss_coef=0., attention_dropout=0.)
    if case.key_repeat:
        config.linear_num_key_heads = 2
        config.linear_num_value_heads = 4
    if case.wrapper:
        config = Qwen3_5MoeConfig(text_config=config.to_dict(),
            vision_config=dict(depth=1, hidden_size=8, intermediate_size=12,
                num_heads=2, out_hidden_size=16, patch_size=2,
                spatial_merge_size=2, temporal_patch_size=1, num_position_embeddings=16))
        config._attn_implementation = "eager"
    return config


def write_source(directory, case):
    directory = Path(directory)
    directory.mkdir(parents=True)
    torch.manual_seed(113)
    cls = TurboQwen3_5MoeForConditionalGeneration if case.wrapper else TurboQwen3_5MoeTextForCausalLM
    model = cls(configuration(case))
    if case.wrapper:
        # Use the same language fixture; vision construction consumes separate RNG draws.
        torch.manual_seed(113)
        language = TurboQwen3_5MoeTextForCausalLM(text_config(num_experts=case.experts))
        model.model.language_model.load_state_dict(language.model.state_dict())
        model.lm_head.load_state_dict(language.lm_head.state_dict())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim > 1:
                parameter.mul_(3)
        for layer in backbone(model, case).layers:
            if layer.layer_type == "linear_attention":
                initialize_gdn(layer.linear_attn)
        if case.empty_owner:
            backbone(model, case).embed_tokens.weight[:, 0] = .5
            for layer in backbone(model, case).layers:
                layer.mlp.gate.weight.zero_()
                layer.mlp.gate.weight[2:, 0] = torch.tensor([.3, .4])
    values = {name: value.clone().contiguous() for name, value in model.state_dict().items()}
    save_file(values, str(directory / "model.safetensors"))
    (directory / "model.safetensors.index.json").write_text(json.dumps(dict(
        metadata=dict(total_size=sum(value.numel() * value.element_size() for value in values.values())),
        weight_map=dict.fromkeys(values, "model.safetensors"))))
    model.config.save_pretrained(directory)


@contextmanager
def cpu_bridge(events=None):
    """Replace unavailable kernel leaves, retaining real runtime bindings."""
    from tests.models.modules.attention.tp_helpers import install_cpu_attention_bridge

    monkeypatch = pytest.MonkeyPatch()
    install_cpu_leaves(monkeypatch, events=events)
    classes = (TurboQwen3_5MoeTextForCausalLM, TurboQwen3_5MoeForConditionalGeneration)
    originals = [cls.init_context for cls in classes]

    def eager_context(original):
        @contextmanager
        def context(cls, config, engine_config):
            with original(config, engine_config):
                config._attn_implementation = "eager"
                if isinstance(config, Qwen3_5MoeConfig):
                    config.text_config._attn_implementation = "eager"
                    config.vision_config._attn_implementation = "eager"
                yield
        return classmethod(context)

    try:
        for cls, original in zip(classes, originals):
            monkeypatch.setattr(cls, "init_context", eager_context(original))
        install_cpu_attention_bridge(monkeypatch)
        yield
    finally:
        monkeypatch.undo()


def prepare(source, case):
    config = EngineConfig.from_dict(dict(mixed_precision="bf16" if case.bf16 else "fp32",
        sp_size=case.sp, tp_size=case.tp, ep_size=case.ep, etp_size=case.tp,
        freeze_policy=dict(parameter_groups=["vision_tower", "multi_modal_projector"]) if case.wrapper else {}))
    config.loss_chunk.enabled = case.chunk
    config.loss_chunk.chunk_size = 2
    config.mfu.enabled = False
    config.activation_checkpointing.enabled = case.ac != "off"
    config.activation_checkpointing.use_reentrant = case.ac == "reentrant"
    prefix = "model.language_model" if case.wrapper else "model"
    config.parallel_plan.plan[prefix + ".layers.*"] = FSDPPlanSpec(activation_checkpoint=case.ac != "off")
    config.moe.aux_loss_coeff, config.moe.z_loss_coeff = case.aux, case.z
    config.checkpoint.hf_export = "final"
    engine = TorchTurboEngine(config)
    from torchturbo.models.qwen3_5_moe import parallel_plan
    from torchturbo.models.qwen3_5_moe_text import parallel_plan as text_plan
    original_replace = parallel_plan._replace_qwen3_5_moe_only
    replacements = []
    def replace(original, *args, **kwargs):
        shared, gate = original.shared_expert, original.shared_expert_gate
        result = original_replace(original, *args, **kwargs)
        assert result.shared_expert is shared and result.shared_expert_gate is gate
        replacements.append(result)
        return result
    with patch.object(parallel_plan, "_replace_qwen3_5_moe_only", replace), patch.object(text_plan, "_replace_qwen3_5_moe_only", replace):
        model = engine.prepare_model(str(source), torch_dtype=torch.float32)
    assert len(replacements) == 2
    expected_cls = TurboQwen3_5MoeForConditionalGeneration if case.wrapper else TurboQwen3_5MoeTextForCausalLM
    assert isinstance(model, expected_cls)
    text = backbone(model, case)
    if case.tp > 1 or case.sp > 1:
        assert text.config._attn_implementation == "turbo_ulysses"
    assert text.rotary_emb.inv_freq.dtype == torch.float32
    assert [layer.layer_type for layer in text.layers] == ["linear_attention", "full_attention"]
    for layer in text.layers:
        assert isinstance(layer.mlp, MoELayer)
        assert isinstance(layer.mlp.experts, GroupedMLP)
        for parameter in layer.mlp.shared_expert.parameters():
            assert not engine.backend._parameter_ownership.is_expert(parameter)
    assert all(not value.is_meta for value in list(model.parameters()) + list(model.buffers()))
    if case.wrapper:
        assert all(not parameter.requires_grad for parameter in model.model.visual.parameters())
        assert model.config.vision_config._attn_implementation == "eager"
    optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=.003, betas=(.8, .95),
        weight_decay=.03, eps=.1 if case.bf16 else 1e-6, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
    return engine, model, optimizer, scheduler


class IndependentMoE(nn.Module):
    """Ordinary expert GEMMs plus the actual native shared expert and gate."""

    def __init__(self, original, config, tp):
        super().__init__()
        hidden, intermediate = config.hidden_size, config.moe_intermediate_size
        self.router = nn.Linear(hidden, config.num_experts, bias=False)
        self.experts = nn.Module()
        shape1 = (config.num_experts * hidden, 2, intermediate) if tp > 1 else (config.num_experts * hidden, 2 * intermediate)
        shape2 = (config.num_experts, intermediate, hidden) if tp > 1 else (config.num_experts * intermediate, hidden)
        self.experts.weight1 = nn.Parameter(torch.empty(shape1))
        self.experts.weight2 = nn.Parameter(torch.empty(shape2))
        self.shared_expert, self.shared_expert_gate = original.shared_expert, original.shared_expert_gate
        self.topk, self.hidden, self.intermediate = config.num_experts_per_tok, hidden, intermediate

    def forward(self, hidden):
        flat = hidden.reshape(-1, self.hidden)
        self.logits = self.router(flat)
        scores = self.logits.float().softmax(-1).to(self.logits.dtype)
        self.ids = scores.topk(self.topk, -1).indices
        mask = torch.zeros_like(scores).scatter(1, self.ids, 1)
        self.probs = scores * mask
        self.probs.retain_grad()
        normalized = self.probs / self.probs.sum(-1, keepdim=True)
        outputs = []
        second = self.experts.weight2.view(-1, self.intermediate, self.hidden)
        for expert, weights in enumerate(self.experts.weight1.view(-1, self.hidden, 2, self.intermediate)):
            gate, up = (flat @ weights.flatten(1)).chunk(2, -1)
            outputs.append((F.silu(gate) * up) @ second[expert])
        self.expert_outputs = [output * normalized[:, expert:expert+1] * mask[:, expert:expert+1]
                               for expert, output in enumerate(outputs)]
        for output in self.expert_outputs:
            output.retain_grad()
        routed = sum(self.expert_outputs)
        shared = self.shared_expert(flat)
        return (routed + shared * self.shared_expert_gate(flat).sigmoid()).reshape_as(hidden), self.logits


def reference_model(source, case):
    cls = TurboQwen3_5MoeForConditionalGeneration if case.wrapper else TurboQwen3_5MoeTextForCausalLM
    model = cls(configuration(case))
    text = backbone(model, case)
    for layer in text.layers:
        layer.mlp = IndependentMoE(layer.mlp, text.config, case.tp)
    values, state = read_hf(source), {}
    for name, parameter in model.named_parameters():
        if name.endswith(".experts.weight1"):
            fused = values[name.removesuffix("weight1") + "gate_up_proj"]
            state[name] = fused.reshape(case.experts, 2, text.config.moe_intermediate_size,
                                        text.config.hidden_size).permute(0, 3, 1, 2).reshape_as(parameter)
        elif name.endswith(".experts.weight2"):
            state[name] = values[name.removesuffix("weight2") + "down_proj"].transpose(1, 2).reshape_as(parameter)
        else:
            state[name] = values[name.replace(".mlp.router.", ".mlp.gate.")]
    model.load_state_dict(state)
    model.on_materialized()
    if case.wrapper:
        model.model.visual.requires_grad_(False)
    plan = OptimizerClassPlan(model, torch.optim.AdamW, {"lr": .003, "weight_decay": .03})
    groups = [group.get_optimizer_param_group() for group in plan.get_param_groups()]
    optimizer = torch.optim.AdamW(groups, lr=.003, betas=(.8, .95), weight_decay=.03,
                                  eps=.1 if case.bf16 else 1e-6, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
    return model, optimizer, scheduler


def observe(model, case):
    stages, inputs, routes, handles = {}, {}, {}, []
    for index, layer in enumerate(backbone(model, case).layers):
        inputs[index], routes[index] = [], []
        def retain(module, args, index=index):
            if torch.is_grad_enabled():
                args[0].retain_grad()
                inputs[index].append(args[0])
        handles.append(layer.register_forward_pre_hook(retain))
        for name in ("input_layernorm", "linear_attn" if index == 0 else "self_attn",
                     "post_attention_layernorm", "mlp", "mlp.shared_expert", "mlp.shared_expert_gate"):
            def stage(module, args, output, key=f"{index}.{name}"):
                value = output[0] if isinstance(output, tuple) else output
                stages[key] = value.detach().clone()
            handles.append(layer.get_submodule(name).register_forward_hook(stage))
        if isinstance(layer.mlp, MoELayer):
            def route(module, args, output, index=index):
                probabilities, routing, logits = output
                if torch.is_grad_enabled():
                    probabilities.retain_grad()
                    routes[index].append((probabilities, routing.detach(), logits.detach()))
            handles.append(layer.mlp.router.register_forward_hook(route))
    handles.append(backbone(model, case).norm.register_forward_hook(
        lambda module, args, output: stages.__setitem__("final", output.detach().clone())))
    return stages, inputs, routes, handles


def segment_attention(module, q, k, v, mask, scaling, **kwargs):
    bounds = module.literal_boundaries
    k = k.repeat_interleave(q.size(1) // k.size(1), 1)
    v = v.repeat_interleave(q.size(1) // v.size(1), 1)
    pieces = [F.scaled_dot_product_attention(q[:, :, start:end], k[:, :, start:end],
        v[:, :, start:end], is_causal=True, dropout_p=0., scale=scaling)
        for start, end in zip(bounds, bounds[1:])]
    output = torch.cat(pieces, 2)
    output = F.pad(output, (0, 0, 0, q.size(2) - bounds[-1]))
    return output.transpose(1, 2).contiguous(), None


def independent_gdn(module, hidden_states, **kwargs):
    return gdn_oracle(module, hidden_states, module.literal_boundaries)


def oracle_step(model, case, step, micros):
    count = sum(int((literal_targets(batch(data, micro, step, bf16=case.bf16),
        batch(data, micro, step, bf16=case.bf16)["labels"].size(1)) != -100).sum())
        for data in range(case.data) for micro in range(micros))
    records, report = {}, torch.tensor(0.)
    for data in range(case.data):
        for micro in range(micros):
            raw = batch(data, micro, step, bf16=case.bf16)
            extent = raw["input_ids"].size(1)
            extent += (-extent) % case.sp
            bounds = raw["cu_seq_lens_q"].tolist() + [extent]
            compute = copy.deepcopy(model) if case.bf16 else model
            if case.bf16:
                compute.zero_grad(set_to_none=True)
                for parameter in compute.parameters():
                    parameter.data = parameter.to(torch.bfloat16)
            text = backbone(compute, case)
            text.layers[0].linear_attn.literal_boundaries = bounds
            text.layers[1].self_attn.literal_boundaries = raw["cu_seq_lens_q"].tolist()
            original_gdn = text.layers[0].linear_attn.forward
            text.layers[0].linear_attn.forward = MethodType(independent_gdn, text.layers[0].linear_attn)
            stages, observed, _, handles = observe(compute, case)
            with patch.object(modeling, "eager_attention_forward", segment_attention):
                output = compute(input_ids=F.pad(raw["input_ids"], (0, extent-raw["input_ids"].size(1))),
                    position_ids=F.pad(raw["position_ids"], (0, extent-raw["input_ids"].size(1))),
                    cu_seq_lens_q=raw["cu_seq_lens_q"], use_cache=False).logits
            text.layers[0].linear_attn.forward = original_gdn
            targets = literal_targets(raw, extent)
            main = sum(F.cross_entropy(logits.reshape(-1, 24), labels.reshape(-1),
                ignore_index=-100, reduction="sum").float() / count
                for logits, labels in zip(output.chunk(case.sp, 1), targets.chunk(case.sp, 1)))
            objective, per_layer = main, {}
            length, local_length = bounds[-2], extent // case.sp
            for index, layer in enumerate(text.layers):
                moe = layer.mlp
                scores = moe.logits[:length].float().softmax(-1)
                counts = F.one_hot(scores.topk(2, -1).indices, case.experts).sum((0, 1)).float()
                aux = (scores.sum(0) * counts).sum() * case.experts / (2 * length * length)
                z = moe.logits[:length].logsumexp(-1).square().mean()
                objective = objective + (case.aux * aux + case.z * z) / (case.data * micros)
                metrics_aux, metrics_z = [], []
                for sp in range(case.sp):
                    start, end = sp * local_length, min((sp + 1) * local_length, length)
                    metrics_aux.append((scores[start:end].sum(0) * counts).sum() * case.experts * case.sp / (2 * length * length))
                    metrics_z.append(moe.logits[start:end].logsumexp(-1).square().sum() * case.sp / length)
                per_layer[index] = dict(probs=moe.probs.detach(), ids=moe.ids.detach(),
                    metric_aux=[value.detach() for value in metrics_aux], metric_z=[value.detach() for value in metrics_z])
            (objective * (case.dense if case.bf16 else 1)).backward()
            for index, layer in enumerate(text.layers):
                per_layer[index].update(input=observed[index][0].grad.detach(),
                    probability_grad=layer.mlp.probs.grad.detach(),
                    expert_objective_active=[bool((output.grad * output.detach()).abs().max() > 0)
                                             for output in layer.mlp.expert_outputs])
            if case.bf16:
                for name, parameter in compute.named_parameters():
                    if parameter.requires_grad:
                        master = model.get_parameter(name)
                        gradient = parameter.grad.float() / case.dense
                        master.grad = gradient if master.grad is None else master.grad + gradient
            for handle in handles:
                handle.remove()
            records[data, micro] = dict(logits=output.detach(), stages=stages, layers=per_layer, main=main.detach())
            report += main.detach()
    return count, report, records


def train_step(engine, model, optimizer, scheduler, reference, ref_optimizer,
               ref_scheduler, case, step, micros, evidence):
    optimizer.zero_grad(set_to_none=True)
    ref_optimizer.zero_grad(set_to_none=True)
    count, expected_report, records = oracle_step(reference, case, step, micros)
    evidence.target_counts.append(dict(step=step, micros=micros, global_count=count,
        local_counts=[sum(int((literal_targets(batch(data, micro, step, bf16=case.bf16),
            batch(data, micro, step, bf16=case.bf16)["labels"].size(1)) != -100).sum()) for micro in range(micros))
            for data in range(case.data)]))
    context = engine.backend._parallel_context
    data, sp = context.data_dp_mesh.get_local_rank(), context.sp_mesh.get_local_rank()
    batches = [batch(data, micro, step, bf16=case.bf16) for micro in range(micros)]
    if case.tp > 1:
        prepared, actual_count = engine.prepare_loss_step(model, batches, torch.device("cpu"))
    else:
        prepared = [engine.prepare_loss_inputs(model, raw) for raw in batches]
        actual_count = count
    assert actual_count == count
    report = torch.tensor(0.)
    metric_before = {name: value.clone() for name, value in engine.metric_tracker.totals.items()}
    counts_before = dict(engine.metric_tracker.counts)
    for micro, inputs in enumerate(prepared):
        stages, observed, routes, handles = observe(model, case)
        with patch.object(modeling, "eager_attention_forward", segment_attention) if case.tp == case.sp == 1 else nullcontext():
            if case.tp == case.sp == 1:
                backbone(model, case).layers[1].self_attn.literal_boundaries = inputs["cu_seq_lens_q"].tolist()
            loss, output = engine.compute_prepared_loss(model, inputs, True,
                num_items_in_batch=count * case.sp if case.tp == 1 else count,
                gradient_accumulation_steps=micros)
        assert not output.main_loss.requires_grad
        report += output.main_loss
        logits = output.logits
        if logits is None:
            logits = F.linear(stages["final"], full(model.lm_head.weight))
        local_length, expected = logits.size(1), records[data, micro]
        begin = sp * local_length
        end = min(begin + local_length, int(inputs["cu_seq_lens_q"][-1]))
        for name, value in stages.items():
            reference_stage = expected["stages"][name]
            value = value.reshape(1, -1, value.size(-1))
            reference_stage = reference_stage.reshape(1, -1, reference_stage.size(-1))
            evidence.close(value[:, :end-begin], reference_stage[:, begin:end], "stage " + name)
        for index in range(2):
            for values in (stages, expected["stages"]):
                shared = values[f"{index}.mlp.shared_expert"]
                gate = values[f"{index}.mlp.shared_expert_gate"]
                values[f"{index}.shared_gated"] = shared * gate.sigmoid()
            actual_shared = stages[f"{index}.shared_gated"].reshape(1, -1, 16)
            expected_shared = expected["stages"][f"{index}.shared_gated"].reshape(1, -1, 16)
            evidence.close(actual_shared[:, :end-begin], expected_shared[:, begin:end], f"shared gated {index}")
            actual_routed = stages[f"{index}.mlp"].reshape(1, -1, 16) - actual_shared
            expected_routed = expected["stages"][f"{index}.mlp"].reshape(1, -1, 16) - expected_shared
            evidence.close(actual_routed[:, :end-begin], expected_routed[:, begin:end], f"routed {index}")
        evidence.close(logits[:, :end-begin], expected["logits"][:, begin:end], "logits")
        loss.backward()
        for index, layer in enumerate(backbone(model, case).layers):
            if case.aux or case.z:
                metrics = layer.mlp.router._metrics
                for metric, coefficient, field in (("aux_loss", case.aux, "metric_aux"), ("z_loss", case.z, "metric_z")):
                    if coefficient:
                        wanted_metric = expected["layers"][index][field][sp]
                        evidence.close(metrics[metric], wanted_metric, f"layer {index} {metric}")
                        evidence.close(metrics[metric + "_scaled"], wanted_metric * coefficient, f"layer {index} {metric} scaled")
        engine.collect_metrics(model)
        scale = 1 if case.bf16 else case.dense
        for index, layer in enumerate(backbone(model, case).layers):
            wanted = expected["layers"][index]
            activations = [value for value in observed[index] if value.grad is not None]
            assert len(activations) == 1
            evidence.close(activations[0].grad, wanted["input"][:, begin:begin+local_length] * scale,
                           f"K scaled input {index}", atol=case.dense * 3e-6)
            active_routes = [entry for entry in routes[index] if entry[0].grad is not None]
            assert len(active_routes) == 1
            probabilities, routing, _ = active_routes[0]
            ids = wanted["ids"][begin:begin+local_length]
            expected_map = F.one_hot(ids, case.experts).sum(1).bool()
            assert torch.equal(routing, expected_map)
            evidence.close(probabilities, wanted["probs"][begin:begin+local_length], f"probabilities {index}")
            evidence.close(probabilities.grad, wanted["probability_grad"][begin:begin+local_length] * scale,
                           f"K scaled probability gradient {index}", atol=case.dense * 3e-6)
            counts = expected_map.sum(0)
            route_record = dict(layer=index, step=step, micro=micro, data=data, sp=sp,
                counts=counts.tolist(), expert_objective_active=wanted["expert_objective_active"])
            if case.ep > 1:
                dispatcher = layer.mlp.token_dispatcher
                wanted_input = counts.reshape(case.ep, case.experts // case.ep).sum(1).tolist()
                assert dispatcher.input_splits.tolist() == wanted_input
                owner = context.ep_mesh.get_local_rank()
                domain = (data * case.sp + sp) // case.ep
                wanted_output = []
                for peer in range(case.ep):
                    peer_data, peer_sp = divmod(domain * case.ep + peer, case.sp)
                    peer_ids = records[peer_data, micro]["layers"][index]["ids"].chunk(case.sp, 0)[peer_sp]
                    peer_counts = F.one_hot(peer_ids, case.experts).sum((0, 1))
                    wanted_output.append(int(peer_counts.chunk(case.ep)[owner].sum()))
                assert dispatcher.output_splits.tolist() == wanted_output
                route_record.update(owner=owner, input_splits=wanted_input, output_splits=wanted_output)
            evidence.routes.append(route_record)
            assert layer.mlp.router.pop_metrics() == {}
        for handle in handles:
            handle.remove()
    if case.tp == 1:
        # Legacy WORLD/SP normalization reports this data replica's contribution.
        expected_report = sum(records[data, micro]["main"] for micro in range(micros)) * case.data
    evidence.close(report, expected_report, "main report")
    for metric, coefficient, field in (("aux_loss", case.aux, "metric_aux"), ("z_loss", case.z, "metric_z")):
        if coefficient:
            expected = sum(records[data, micro]["layers"][index][field][sp]
                           for micro in range(micros) for index in range(2))
            for suffix, scale in (("", 1.), ("_scaled", coefficient)):
                name = metric + suffix
                evidence.close(engine.metric_tracker.totals[name] - metric_before.get(name, 0.), expected * scale, name)
                assert engine.metric_tracker.counts[name] == counts_before.get(name, 0) + 2 * micros
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if case.tp == 1 and case.ep > 1:
        # Legacy EP compensation normally runs inside backend clipping.
        engine.backend._scale_moe_gradient([parameter for name, parameter in parameters if ".experts." in name])
    for name, parameter in parameters:
        assert parameter.grad is not None, name
        assert parameter.grad.to_local().shape == parameter.to_local().shape
        assert not any(isinstance(placement, Partial) for placement in parameter.grad.placements), name
        expected = expected_local(reference.get_parameter(name).grad, name, context)
        assert torch.isfinite(full(parameter.grad)).all(), name
        evidence.gradient(parameter.grad, expected, name)
        if case.empty_owner and context.ep_mesh.get_local_rank() == 0 and ".experts." in name:
            assert torch.count_nonzero(expected) == 0
        else:
            assert expected.abs().max() > 1e-10, name
        if "self_attn.q_proj.weight" in name:
            head_pairs = expected.reshape(4, 2, 8, 16)
            assert all(head_pairs[:, index].norm() > 1e-8 for index in (0, 1))
        if name.endswith(".experts.weight1") and not case.empty_owner:
            hidden = backbone(reference, case).config.hidden_size
            intermediate = backbone(reference, case).config.moe_intermediate_size
            magnitudes = [float(block.norm()) for block in expected.reshape(-1, hidden, 2, intermediate)]
            evidence.expert_magnitudes.setdefault(name, []).append(magnitudes)
    norm = torch.linalg.vector_norm(torch.cat([parameter.grad.flatten() for parameter in reference.parameters() if parameter.requires_grad]))
    if case.clip:
        assert norm > .15
        evidence.close(engine.backend.clip_grad_norm_(model.parameters(), .15), norm, "norm")
        torch.nn.utils.clip_grad_norm_(reference.parameters(), .15)
        for name, parameter in parameters:
            evidence.close(parameter.grad, expected_local(reference.get_parameter(name).grad, name, context), "clipped " + name)
    else:
        dense_square, expert_square = torch.tensor(0.), torch.tensor(0.)
        for name, parameter in parameters:
            square = full(parameter.grad).square().sum()
            if ".experts." in name:
                expert_square += square
            else:
                dense_square += square
        dist.all_reduce(expert_square, group=context.ep_mesh.get_group())
        evidence.close((dense_square + expert_square).sqrt(), norm, "norm")
    before = {name: full(parameter) for name, parameter in parameters}
    ref_before = {name: reference.get_parameter(name).detach().clone() for name, _ in parameters}
    optimizer.step()
    ref_optimizer.step()
    scheduler.step()
    ref_scheduler.step()
    for name, parameter in parameters:
        ref_parameter = reference.get_parameter(name)
        evidence.close(parameter, expected_local(ref_parameter, name, context), "parameter " + name)
        evidence.close(full(parameter) - before[name], expected_local(ref_parameter.detach() - ref_before[name], name, context), "update " + name)
        for field, value in optimizer.state[parameter].items():
            assert torch.isfinite(full(value)).all(), (name, field)
            wanted = ref_optimizer.state[ref_parameter][field]
            if field != "step":
                wanted = expected_local(wanted, name, context)
                assert value.dtype == torch.float32
            evidence.close(value, wanted, "optimizer " + name + " " + field)
        if case.empty_owner and ".experts." in name:
            owner_record = dict(owner=context.ep_mesh.get_local_rank(), parameter=name,
                gradient_nonzero=int(torch.count_nonzero(full(parameter.grad))),
                exp_avg_nonzero=int(torch.count_nonzero(full(optimizer.state[parameter]["exp_avg"]))),
                exp_avg_sq_nonzero=int(torch.count_nonzero(full(optimizer.state[parameter]["exp_avg_sq"]))))
            if context.ep_mesh.get_local_rank() == 0:
                assert all(owner_record[field] == 0 for field in ("gradient_nonzero", "exp_avg_nonzero", "exp_avg_sq_nonzero"))
                assert (full(parameter) - before[name]).norm() > 0
            else:
                assert all(owner_record[field] > 0 for field in ("gradient_nonzero", "exp_avg_nonzero", "exp_avg_sq_nonzero"))
            evidence.empty_owners.append(owner_record)
    assert scheduler.last_epoch == ref_scheduler.last_epoch
    for left, right in ((scheduler, scheduler.base_lrs), (ref_scheduler, ref_scheduler.base_lrs)):
        assert left.get_last_lr() == pytest.approx([lr * .8 ** left.last_epoch for lr in right])
    return float(norm)
