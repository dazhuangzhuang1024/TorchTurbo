"""CPU construction/attention bridges and independent full-Qwen oracle."""

import copy
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe import modeling_qwen3_moe as hf_qwen

from torchturbo.engine import TorchTurboEngine
from torchturbo.engine.config import EngineConfig
from torchturbo.engine.optimizers import OptimizerClassPlan
from torchturbo.models.qwen3_moe.model import TurboQwen3MoeForCausalLM
from torchturbo.ops import get_or_create_ops
from torchturbo.parallel import FSDPPlanSpec
from torchturbo.runtime.cpu import CpuRuntime
from tests.engine.backends.fsdp2.etp_checkpoint_helpers import read_hf


@dataclass(frozen=True)
class Case:
    world: int = 4
    sp: int = 1
    ep: int = 1
    experts: int = 4
    aux: float = 0.
    z: float = 0.
    ac: str = "off"
    clip: bool = False
    chunk: bool = False
    tied: bool = False
    bf16: bool = False
    empty_owner: bool = False

    @property
    def data(self):
        return self.world // (2 * self.sp)

    @property
    def dense(self):
        return self.world // 2


def qwen_config(case):
    config = Qwen3MoeConfig(vocab_size=24, hidden_size=16, intermediate_size=12,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        head_dim=4, max_position_embeddings=32, moe_intermediate_size=12,
        num_experts=case.experts, num_experts_per_tok=2, attention_dropout=0.,
        use_cache=False, output_router_logits=False, router_aux_loss_coef=0.,
        tie_word_embeddings=case.tied, dtype=torch.float32)
    config._attn_implementation = "eager"
    return config


def write_qwen_source(directory, case):
    directory = Path(directory)
    directory.mkdir(parents=True)
    torch.manual_seed(113)
    model = hf_qwen.Qwen3MoeForCausalLM(qwen_config(case))
    # Larger explicit source matrices make small router/expert gradients useful
    # numerical witnesses. Norm scales retain the actual model initialization.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim > 1:
                parameter.mul_(3)
        if case.empty_owner:
            # A positive coordinate gives experts 2/3 separated router logits.
            # This exercises an empty owner without a top-k tie boundary.
            model.model.embed_tokens.weight[:, 0] = .5
            router = model.model.layers[0].mlp.gate.weight
            router.zero_()
            router[2:, 0] = torch.tensor([.3, .4])
    values = {}
    for name, value in model.state_dict().items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix(".gate_up_proj")
            for expert, weights in enumerate(value):
                for projection, weight in zip(("gate_proj", "up_proj"), weights.chunk(2)):
                    values[f"{prefix}.{expert}.{projection}.weight"] = weight.clone().contiguous()
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix(".down_proj")
            for expert, weight in enumerate(value):
                values[f"{prefix}.{expert}.down_proj.weight"] = weight.clone().contiguous()
        elif not (case.tied and name == "lm_head.weight"):
            values[name] = value.clone().contiguous()
    save_file(values, str(directory / "model.safetensors"))
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in values.values())},
        "weight_map": dict.fromkeys(values, "model.safetensors"),
    }))
    model.config.save_pretrained(directory)


@contextmanager
def cpu_qwen_bridge():
    """Preserve real engine setup and use its existing native CPU attention."""
    from torchturbo.models.modules.attention.ulysses import ulysses
    from torchturbo.models.modules.attention.interface import attention_forward

    original_context = TurboQwen3MoeForCausalLM.init_context

    @contextmanager
    def eager_construction(cls, config, engine_config):
        with original_context(config, engine_config):
            config._attn_implementation = "eager"
            yield

    def attention(*args, **kwargs):
        assert not kwargs.pop("return_max_logits", False)
        output, _ = attention_forward(*args, return_max_logits=True, **kwargs)
        return output

    with patch.object(TurboQwen3MoeForCausalLM, "init_context", classmethod(eager_construction)), \
            patch.object(ulysses, "attention_forward", attention):
        yield


def prepare_qwen(source, case):
    config = EngineConfig.from_dict(dict(mixed_precision="bf16" if case.bf16 else "fp32",
        sp_size=case.sp, tp_size=2, ep_size=case.ep, etp_size=2))
    config.loss_chunk.enabled = case.chunk
    config.loss_chunk.chunk_size = 2
    config.mfu.enabled = False
    config.activation_checkpointing.enabled = case.ac != "off"
    config.activation_checkpointing.use_reentrant = case.ac == "reentrant"
    config.parallel_plan.plan["model.layers.*"] = FSDPPlanSpec(activation_checkpoint=case.ac != "off")
    config.moe.aux_loss_coeff = case.aux
    config.moe.z_loss_coeff = case.z
    config.checkpoint.hf_export = "final"
    engine = TorchTurboEngine(config)
    model = engine.prepare_model(str(source), torch_dtype=torch.float32)
    assert isinstance(model, TurboQwen3MoeForCausalLM)
    assert isinstance(engine.device_runtime, CpuRuntime)
    assert engine.ops is get_or_create_ops(device_runtime=CpuRuntime())
    assert model.config._attn_implementation == "turbo_ulysses"
    assert model.model.rotary_emb.inv_freq.dtype == torch.bfloat16
    assert model.model.layers[0].mlp.router.config.moe_router_dtype is None
    assert all(not value.is_meta for value in list(model.parameters()) + list(model.buffers()))
    if case.tied:
        assert model.lm_head.weight is model.model.embed_tokens.weight
    optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=.003, betas=(.8, .95),
        weight_decay=.03, eps=.1 if case.bf16 else 1e-6, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
    return engine, model, optimizer, scheduler


def batch(data, micro, step, *, bf16=False):
    # Distinct rows have literal, different packed lengths and a physical tail.
    boundaries = (0, 1, 3, 6) if data % 2 == 0 else (0, 2, 6, 8)
    length = boundaries[-1] + 1
    ids = (torch.arange(length).unsqueeze(0) + 1 + data * 5 + step * 3 + micro) % 24
    labels = ids.clone()
    if step == 0 and (micro == 1 or (data == 1 and not bf16)):
        labels.fill_(-100)
    elif data % 2:
        labels[0, 4] = -100
    positions = [index - start for start, end in zip(boundaries[:-1], boundaries[1:])
                 for index in range(start, end)] + [0]
    cu = torch.tensor(boundaries, dtype=torch.int32)
    return dict(input_ids=ids, labels=labels, position_ids=torch.tensor([positions]),
        cu_seq_lens_q=cu, cu_seq_lens_k=cu.clone(),
        max_length_q=max(b-a for a, b in zip(boundaries[:-1], boundaries[1:])),
        max_length_k=max(b-a for a, b in zip(boundaries[:-1], boundaries[1:])))


def literal_targets(inputs, extent):
    labels = inputs["labels"]
    result = torch.full((1, extent), -100, dtype=torch.long)
    # Deliberately no production target helper or bucketized target construction.
    bounds = inputs["cu_seq_lens_q"].tolist()
    for first, stop in zip(bounds[:-1], bounds[1:]):
        for position in range(first, stop - 1):
            result[0, position] = labels[0, position + 1]
    return result


def segment_sdpa(module, query, key, value, attention_mask, scaling, **kwargs):
    bounds = kwargs["cu_seq_lens_q"].tolist()
    pieces = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        pieces.append(F.scaled_dot_product_attention(query[:, :, start:end],
            key[:, :, start:end], value[:, :, start:end], is_causal=True,
            dropout_p=0., scale=scaling))
    output = torch.cat(pieces, dim=2)
    output = F.pad(output, (0, 0, 0, query.size(2) - bounds[-1]))
    return output.transpose(1, 2).contiguous(), None


class IndependentExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(config.num_experts * 16, 2, 12))
        self.weight2 = nn.Parameter(torch.empty(config.num_experts, 12, 16))


class IndependentMoE(nn.Module):
    """Ordinary per-expert GEMMs with explicit probability multiplication."""

    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(16, config.num_experts, bias=False)
        self.experts = IndependentExperts(config)

    def forward(self, hidden):
        flat = hidden.reshape(-1, 16)
        self.logits = self.router(flat)
        self.scores = self.logits.float().softmax(-1).to(self.logits.dtype)
        self.ids = self.scores.topk(2, dim=-1).indices
        mask = torch.zeros_like(self.scores).scatter(1, self.ids, 1)
        self.probs = self.scores * mask
        self.probs.retain_grad()
        normalized_probs = self.probs / self.probs.sum(-1, keepdim=True)
        outputs = []
        for expert, weights in enumerate(self.experts.weight1.view(-1, 16, 2, 12)):
            gate, up = (flat @ weights.flatten(1)).chunk(2, dim=-1)
            outputs.append((F.silu(gate) * up) @ self.experts.weight2[expert])
        # Unselected probabilities have zero gradient, matching route selection.
        combined = sum(output * normalized_probs[:, expert:expert+1] * mask[:, expert:expert+1]
                       for expert, output in enumerate(outputs))
        return combined.reshape_as(hidden)


def reference_model(source, case):
    model = TurboQwen3MoeForCausalLM(qwen_config(case))
    model.model.layers[0].mlp = IndependentMoE(model.config)
    values = read_hf(source)
    state = {}
    for name, parameter in model.named_parameters():
        prefix = "model.layers.0.mlp.experts"
        if name == prefix + ".weight1":
            state[name] = torch.stack([torch.stack([
                values[f"{prefix}.{expert}.{projection}.weight"].t()
                for projection in ("gate_proj", "up_proj")], dim=1)
                for expert in range(case.experts)]).reshape_as(parameter)
        elif name == prefix + ".weight2":
            state[name] = torch.stack([values[f"{prefix}.{expert}.down_proj.weight"].t()
                                      for expert in range(case.experts)])
        else:
            state[name] = values[name.replace(".mlp.router.", ".mlp.gate.")]
    if case.tied:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state)
    model.on_materialized()
    plan = OptimizerClassPlan(model, torch.optim.AdamW, {"lr": .003, "weight_decay": .03})
    groups = [group.get_optimizer_param_group() for group in plan.get_param_groups()]
    optimizer = torch.optim.AdamW(groups, lr=.003, betas=(.8, .95), weight_decay=.03,
                                  eps=.1 if case.bf16 else 1e-6, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
    return model, optimizer, scheduler


def full(value):
    if isinstance(value, DTensor):
        value = value.full_tensor()
    return value.detach().clone()


def expected_local(value, name, context):
    if ".experts." not in name:
        return value
    index = context.ep_mesh.get_local_rank()
    return value.chunk(context.ep_size, dim=0)[index]


def retain_input(module, args):
    if torch.is_grad_enabled():
        args[0].retain_grad()
        module.observed_inputs.append(args[0])


def install_observers(model):
    layer = model.model.layers[0]
    layer.observed_inputs = []
    layer.register_forward_pre_hook(retain_input)
    model.route_records = []
    model.last_hidden = None

    def router_hook(module, args, output):
        probabilities, routing, logits = output
        if torch.is_grad_enabled():
            probabilities.retain_grad()
            model.route_records.append((probabilities, routing.detach(), logits.detach()))

    layer.mlp.router.register_forward_hook(router_hook)
    model.model.norm.register_forward_hook(lambda module, args, output: setattr(model, "last_hidden", output))


def observe_stages(model):
    values, handles = {}, []
    for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
        module = model.model.layers[0].get_submodule(name)
        def record(module, args, output, name=name):
            values[name] = (output[0] if isinstance(output, tuple) else output).detach().clone()
        handles.append(module.register_forward_hook(record))
    return values, handles


def oracle_step(model, case, step, micros):
    count = sum(int((literal_targets(batch(data, micro, step, bf16=case.bf16),
                    batch(data, micro, step, bf16=case.bf16)["labels"].size(1)) != -100).sum())
                for data in range(case.data) for micro in range(micros))
    records, total_report = {}, torch.tensor(0.)
    for data in range(case.data):
        for micro in range(micros):
            inputs = batch(data, micro, step, bf16=case.bf16)
            extent = inputs["input_ids"].size(1)
            extent += (-extent) % case.sp
            targets = literal_targets(inputs, extent)
            compute = copy.deepcopy(model) if case.bf16 else model
            if case.bf16:
                compute.zero_grad(set_to_none=True)
                for parameter in compute.parameters():
                    parameter.data = parameter.detach().to(torch.bfloat16)
            layer = compute.model.layers[0]
            stages, stage_handles = observe_stages(compute)
            layer.observed_inputs = []
            handle = layer.register_forward_pre_hook(retain_input)
            with patch.object(hf_qwen, "eager_attention_forward", segment_sdpa):
                output = compute(input_ids=F.pad(inputs["input_ids"], (0, extent-inputs["input_ids"].size(1))),
                    position_ids=F.pad(inputs["position_ids"], (0, extent-inputs["position_ids"].size(1))),
                    cu_seq_lens_q=inputs["cu_seq_lens_q"], use_cache=False).logits
            handle.remove()
            for stage_handle in stage_handles:
                stage_handle.remove()
            sums = [F.cross_entropy(logits.reshape(-1, 24), labels.reshape(-1),
                        ignore_index=-100, reduction="sum")
                    for logits, labels in zip(output.chunk(case.sp, dim=1), targets.chunk(case.sp, dim=1))]
            main = sum(value.float() / count for value in sums)
            moe = layer.mlp
            length = int(inputs["cu_seq_lens_q"][-1])
            scores = moe.logits[:length].float().softmax(-1)
            ids = scores.topk(2, dim=-1).indices
            counts = F.one_hot(ids, case.experts).sum((0, 1)).float()
            raw_aux = (scores.sum(0) * counts).sum() * case.experts / (2 * length * length)
            raw_z = moe.logits[:length].logsumexp(-1).square().mean()
            metric_aux, metric_z = [], []
            local_length = extent // case.sp
            for sp in range(case.sp):
                start, end = sp * local_length, min((sp + 1) * local_length, length)
                metric_aux.append((scores[start:end].sum(0) * counts).sum()
                    * case.experts * case.sp / (2 * length * length))
                metric_z.append(moe.logits[start:end].logsumexp(-1).square().sum() * case.sp / length)
            objective = main + (case.aux * raw_aux + case.z * raw_z) / (case.data * micros)
            (objective * (case.dense if case.bf16 else 1)).backward()
            if case.bf16:
                for name, parameter in compute.named_parameters():
                    master = model.get_parameter(name)
                    gradient = parameter.grad.float() / case.dense
                    master.grad = gradient if master.grad is None else master.grad + gradient
            records[data, micro] = dict(logits=output.detach(), input=layer.observed_inputs[0].grad.detach(),
                probs=moe.probs.detach(), probability_grad=moe.probs.grad.detach(), ids=moe.ids.detach(),
                raw_aux=raw_aux.detach(), raw_z=raw_z.detach(), stages=stages,
                metric_aux=[value.detach() for value in metric_aux],
                metric_z=[value.detach() for value in metric_z])
            total_report += main.detach()
    return count, total_report, records


class Evidence:
    def __init__(self):
        self.errors = {}
        self.scales = {}
        self.routes = []
        self.empty_owners = []

    def close(self, actual, expected, name, *, rtol=3e-5, atol=3e-6):
        actual, expected = full(actual).float(), full(expected).float()
        error = float((actual - expected).abs().max()) if actual.numel() else 0.
        self.errors[name] = max(self.errors.get(name, 0.), error)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol,
            msg=lambda message: f"{name}: max_error={error}\n{message}")

    def gradient(self, actual, expected, name):
        self.close(actual, expected, "gradient " + name)
        expected = expected.double()
        denominator = expected.square().sum()
        if denominator > 0:
            scale = float((full(actual).double() * expected).sum() / denominator)
            self.scales.setdefault(name, []).append(scale)
            assert .995 < scale < 1.005, (name, scale)


def train_step(engine, model, optimizer, scheduler, reference, ref_optimizer,
               ref_scheduler, case, step, micros, evidence):
    optimizer.zero_grad(set_to_none=True)
    ref_optimizer.zero_grad(set_to_none=True)
    count, expected_report, records = oracle_step(reference, case, step, micros)
    context = engine.backend._parallel_context
    data, sp = context.data_dp_mesh.get_local_rank(), context.sp_mesh.get_local_rank()
    prepared, actual_count = engine.prepare_loss_step(model,
        [batch(data, micro, step, bf16=case.bf16) for micro in range(micros)], torch.device("cpu"))
    assert actual_count == count
    report = torch.tensor(0.)
    metric_before = {name: value.clone() for name, value in engine.metric_tracker.totals.items()}
    counts_before = dict(engine.metric_tracker.counts)
    for micro, inputs in enumerate(prepared):
        stages, stage_handles = observe_stages(model)
        model.route_records.clear()
        model.model.layers[0].observed_inputs.clear()
        loss, output = engine.compute_prepared_loss(model, inputs, True,
            num_items_in_batch=count, gradient_accumulation_steps=micros)
        expected = records[data, micro]
        assert not output.main_loss.requires_grad
        report += output.main_loss
        logits = output.logits
        if logits is None:
            logits = F.linear(model.last_hidden.detach(), full(model.lm_head.weight))
        local_length = logits.size(1)
        begin = sp * local_length
        end = min(begin + local_length, int(inputs["cu_seq_lens_q"][-1]))
        if model.route_records:
            evidence.close(model.route_records[0][0], expected["probs"][begin:begin+local_length], "forward probabilities")
        for name, value in stages.items():
            evidence.close(value[:, :end-begin], expected["stages"][name][:, begin:end], "stage " + name)
        for handle in stage_handles:
            handle.remove()
        evidence.close(logits[:, :end-begin], expected["logits"][:, begin:end], "logits")
        loss.backward()
        engine.collect_metrics(model)
        activation = [value for value in model.model.layers[0].observed_inputs if value.grad is not None]
        assert len(activation) == 1
        activation_scale = 1 if case.bf16 else case.dense
        evidence.close(activation[0].grad, expected["input"][:, begin:begin+local_length] * activation_scale,
                       "K scaled input", atol=case.dense * 3e-6)
        evidence.errors["logical input"] = evidence.errors["K scaled input"] / case.dense
        routes = [record for record in model.route_records if record[0].grad is not None]
        assert len(routes) == 1
        probs, routing, _ = routes[0]
        expected_ids = expected["ids"][begin:begin+local_length]
        expected_map = F.one_hot(expected_ids, case.experts).sum(1).bool()
        assert torch.equal(routing, expected_map)
        counts = expected_map.sum(0)
        dispatcher = model.model.layers[0].mlp.token_dispatcher
        route_record = dict(step=step, micro=micro, data=data, sp=sp,
                            counts=counts.tolist(), ids=expected_ids.tolist())
        if case.ep > 1:
            local_experts = case.experts // case.ep
            wanted_input = counts.reshape(case.ep, local_experts).sum(1).tolist()
            assert dispatcher.input_splits.tolist() == wanted_input
            owner = context.ep_mesh.get_local_rank()
            domain = (data * case.sp + sp) // case.ep
            wanted_output = []
            for peer in range(case.ep):
                dense_rank = domain * case.ep + peer
                peer_data, peer_sp = divmod(dense_rank, case.sp)
                peer_ids = records[peer_data, micro]["ids"].chunk(case.sp, dim=0)[peer_sp]
                peer_counts = F.one_hot(peer_ids, case.experts).sum((0, 1))
                wanted_output.append(int(peer_counts[owner*local_experts:(owner+1)*local_experts].sum()))
            assert dispatcher.output_splits.tolist() == wanted_output
            route_record.update(owner=owner, input_splits=wanted_input, output_splits=wanted_output)
            if case.empty_owner and step == 0:
                assert counts[:2].sum() == 0
                assert counts[2:].sum() == expected_ids.numel()
                assert sum(wanted_output) == (0 if owner == 0 else expected_ids.numel() * case.ep)
        evidence.routes.append(route_record)
        evidence.close(probs, expected["probs"][begin:begin+local_length], "probabilities")
        evidence.close(probs.grad, expected["probability_grad"][begin:begin+local_length] * activation_scale,
                       "K scaled probability gradient", atol=case.dense * 3e-6)
        evidence.errors["logical probability gradient"] = evidence.errors["K scaled probability gradient"] / case.dense
    evidence.close(report, expected_report, "main report")
    for metric, coefficient, field in (("aux_loss", case.aux, "metric_aux"),
                                       ("z_loss", case.z, "metric_z")):
        if coefficient:
            expected = sum(records[data, micro][field][sp] for micro in range(micros))
            for suffix, scale in (("", 1.), ("_scaled", coefficient)):
                name = metric + suffix
                actual = engine.metric_tracker.totals[name] - metric_before.get(name, 0.)
                evidence.close(actual, expected * scale, name)
                assert engine.metric_tracker.counts[name] == counts_before.get(name, 0) + micros
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.to_local().shape == parameter.to_local().shape
        expected = expected_local(reference.get_parameter(name).grad, name, context)
        evidence.gradient(parameter.grad, expected, name)
        if case.empty_owner and step == 0 and context.ep_mesh.get_local_rank() == 0 and ".experts." in name:
            assert torch.count_nonzero(expected) == 0
            assert torch.count_nonzero(full(parameter.grad)) == 0
    norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in reference.parameters()]))
    if case.clip:
        assert norm > .15
        evidence.close(engine.backend.clip_grad_norm_(model.parameters(), .15), norm, "norm")
        torch.nn.utils.clip_grad_norm_(reference.parameters(), .15)
        for name, parameter in model.named_parameters():
            evidence.close(parameter.grad, expected_local(reference.get_parameter(name).grad, name, context),
                           "clipped " + name)
    else:
        # Inspect the norm without invoking backend clipping in the no-clip case.
        dense_square, expert_square = torch.tensor(0.), torch.tensor(0.)
        for name, parameter in model.named_parameters():
            square = full(parameter.grad).square().sum()
            if ".experts." in name:
                expert_square += square
            else:
                dense_square += square
        dist.all_reduce(expert_square, group=context.ep_mesh.get_group())
        evidence.close((dense_square + expert_square).sqrt(), norm, "norm")
    before = {name: full(parameter) for name, parameter in model.named_parameters()}
    ref_before = {name: parameter.detach().clone() for name, parameter in reference.named_parameters()}
    optimizer.step()
    ref_optimizer.step()
    scheduler.step()
    ref_scheduler.step()
    for name, parameter in model.named_parameters():
        expected_parameter = reference.get_parameter(name)
        evidence.close(parameter, expected_local(expected_parameter, name, context), "parameter " + name)
        evidence.close(full(parameter) - before[name], expected_local(expected_parameter.detach() - ref_before[name], name, context),
                       "update " + name)
        for field, value in optimizer.state[parameter].items():
            expected = ref_optimizer.state[expected_parameter][field]
            if field != "step":
                expected = expected_local(expected, name, context)
            evidence.close(value, expected, "optimizer " + name + " " + field)
            if case.empty_owner and step == 0 and context.ep_mesh.get_local_rank() == 0 and ".experts." in name:
                if field == "step":
                    assert value == 1
                else:
                    assert torch.count_nonzero(full(value)) == 0
        if case.empty_owner and step == 0 and ".experts." in name:
            evidence.empty_owners.append(dict(owner=context.ep_mesh.get_local_rank(), parameter=name,
                gradient_nonzero=int(torch.count_nonzero(full(parameter.grad))),
                exp_avg_nonzero=int(torch.count_nonzero(full(optimizer.state[parameter]["exp_avg"]))),
                exp_avg_sq_nonzero=int(torch.count_nonzero(full(optimizer.state[parameter]["exp_avg_sq"])))))
    return float(norm)
