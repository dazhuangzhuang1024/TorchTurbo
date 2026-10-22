"""Independent E>1 shared-expert/gate acceptance, separate from Qwen."""

import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from torchturbo.engine import TorchTurboEngine
from torchturbo.engine.config import EngineConfig
from torchturbo.engine.optimizers import OptimizerClassPlan
from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext
from tests.engine.backends.fsdp2.etp_checkpoint_helpers import expert_values
from tests.engine.backends.fsdp2.test_etp_training import SharedMoETrainingModel, _plan
from tests.models.qwen3_moe.etp_helpers import Evidence, expected_local, full


class SharedReference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.moe = nn.Module()
        self.moe.router = nn.Linear(3, 4, bias=False)
        self.moe.router.weight = nn.Parameter(model.moe.router.weight.detach().clone())
        self.moe.experts = nn.Module()
        w1, w2 = expert_values(4)
        self.moe.experts.weight1 = nn.Parameter(w1.reshape(12, 2, 4))
        self.moe.experts.weight2 = nn.Parameter(w2)
        self.moe.shared_expert = copy.deepcopy(model.moe.shared_expert)
        self.moe.shared_expert_gate = copy.deepcopy(model.moe.shared_expert_gate)
        self.norm, self.head = copy.deepcopy(model.norm), copy.deepcopy(model.head)

    def forward(self, x, padding):
        logits = self.moe.router(x).reshape(-1, 4)
        scores = logits.sigmoid()
        ids = (scores + torch.tensor([0., .002, -.001, .001])).topk(2, dim=-1).indices
        selected = scores.gather(1, ids)
        selected = selected / selected.sum(1, keepdim=True)
        self.probs = torch.zeros_like(scores).scatter(1, ids, selected)
        self.probs.retain_grad()
        mask = F.one_hot(ids, 4).sum(1)
        outputs = []
        for expert, weight in enumerate(self.moe.experts.weight1.view(4, 3, 8)):
            gate, up = (x.reshape(-1, 3) @ weight).chunk(2, -1)
            outputs.append((F.silu(gate) * up) @ self.moe.experts.weight2[expert])
        routed = sum(value * self.probs[:, expert:expert+1] * mask[:, expert:expert+1]
                     for expert, value in enumerate(outputs)).reshape_as(x)
        shared = self.moe.shared_expert(x) * self.moe.shared_expert_gate(x).sigmoid()
        valid_logits = logits[~padding.flatten()]
        aux_scores = valid_logits.sigmoid()
        aux_scores = aux_scores / aux_scores.sum(1, keepdim=True)
        counts = F.one_hot(aux_scores.topk(2, dim=-1).indices, 4).sum((0, 1))
        aux = (aux_scores.sum(0) * counts).sum() * 4 / (2 * len(valid_logits) ** 2)
        z = valid_logits.logsumexp(-1).square().mean()
        return self.head(self.norm(routed + shared)), aux, z


def shared_inputs(data, step, micro):
    x = (torch.arange(24.).view(4, 2, 3) / 19 - .6 + data * .11 + step * .13 + micro * .07)
    padding = torch.zeros(4, 2, dtype=torch.bool)
    padding[-1, -1] = True
    return x.requires_grad_(), padding


def shared_worker(rank, directory, sp):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous",
        rank=rank, world_size=4 * sp, timeout=timedelta(seconds=120))
    try:
        context = ParallelContext("cpu", sp_size=sp, tp_size=2, ep_size=2, etp_size=2)
        config = EngineConfig.from_dict(dict(mixed_precision="fp32", sp_size=sp,
            tp_size=2, ep_size=2, etp_size=2))
        config.moe.aux_loss_coeff, config.moe.z_loss_coeff = .05, .03
        config.activation_checkpointing.enabled = True
        config.activation_checkpointing.use_reentrant = False
        config.mfu.enabled = False
        engine = TorchTurboEngine(config)
        model = SharedMoETrainingModel(context.ep_mesh.get_group(), num_moe_experts=4,
            moe_router_topk=2, moe_router_enable_expert_bias=True)
        with torch.no_grad():
            model.moe.router.expert_bias.copy_(torch.tensor([0., .002, -.001, .001]))
            w1, w2 = expert_values(4)
            ep = context.ep_mesh.get_local_rank()
            model.moe.experts.weight1.copy_(w1[ep*2:ep*2+2].reshape(6, 8))
            model.moe.experts.weight2.copy_(w2[ep*2:ep*2+2].reshape(8, 3))
        reference = SharedReference(model)
        engine.backend.prepare_model(model, _plan(context, checkpoint=True))
        context = engine.backend._parallel_context
        engine.model = model
        optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=.003, weight_decay=.03, eps=1e-6)
        ref_plan = OptimizerClassPlan(reference, torch.optim.AdamW, {"lr": .003, "weight_decay": .03})
        groups = [group.get_optimizer_param_group() for group in ref_plan.get_param_groups()]
        ref_optimizer = torch.optim.AdamW(groups, lr=.003, weight_decay=.03, eps=1e-6)
        probabilities = []
        def observe(module, args, output):
            output[0].retain_grad()
            probabilities.append(output[0])
        model.moe.router.register_forward_hook(observe)
        evidence, norms = Evidence(), []
        for step, micros in enumerate((2, 1)):
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            expected = {}
            for data in range(2):
                for micro in range(micros):
                    x, padding = shared_inputs(data, step, micro)
                    output, aux, z = reference(x, padding)
                    objective = (output.masked_fill(padding[..., None], 0).square().sum() + .05*aux + .03*z) / (2*micros)
                    objective.backward()
                    expected[data, micro] = (output.detach(), x.grad, reference.probs.detach(), reference.probs.grad)
            data, sp_rank = context.data_dp_mesh.get_local_rank(), context.sp_mesh.get_local_rank()
            for micro in range(micros):
                probabilities.clear()
                x, padding = shared_inputs(data, step, micro)
                engine._prepare_moe_runtime(model, {"padding_mask": padding}, micros)
                output = model(x=x)
                mask = padding.chunk(sp, dim=0)[sp_rank]
                (output.masked_fill(mask[..., None], 0).square().sum() * sp / micros).backward()
                engine.collect_metrics(model)
                ref_output, ref_input, ref_probs, ref_prob_grad = expected[data, micro]
                evidence.close(output, ref_output.chunk(sp, dim=0)[sp_rank], "shared output")
                wanted_input = torch.zeros_like(x)
                wanted_input.chunk(sp, dim=0)[sp_rank].copy_(ref_input.chunk(sp, dim=0)[sp_rank] * context.dense_fsdp_mesh.size())
                evidence.close(x.grad, wanted_input, "shared K input")
                active = [value for value in probabilities if value.grad is not None]
                assert len(active) == 1
                evidence.close(active[0], ref_probs.chunk(sp, dim=0)[sp_rank], "shared probability")
                evidence.close(active[0].grad, ref_prob_grad.chunk(sp, dim=0)[sp_rank] * context.dense_fsdp_mesh.size(),
                               "shared K probability gradient")
            for name, parameter in model.named_parameters():
                evidence.gradient(parameter.grad, expected_local(reference.get_parameter(name).grad, name, context), name)
            norm = torch.linalg.vector_norm(torch.cat([parameter.grad.flatten() for parameter in reference.parameters()]))
            assert norm > .03
            norms.append(float(norm))
            evidence.close(engine.backend.clip_grad_norm_(model.parameters(), .03), norm, "shared norm")
            torch.nn.utils.clip_grad_norm_(reference.parameters(), .03)
            for name, parameter in model.named_parameters():
                evidence.close(parameter.grad, expected_local(reference.get_parameter(name).grad, name, context), "shared clipped " + name)
            optimizer.step()
            ref_optimizer.step()
            for name, parameter in model.named_parameters():
                reference_parameter = reference.get_parameter(name)
                evidence.close(parameter, expected_local(reference_parameter, name, context), "shared parameter " + name)
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    wanted = ref_optimizer.state[reference_parameter][field]
                    if field != "step":
                        wanted = expected_local(wanted, name, context)
                    evidence.close(optimizer.state[parameter][field], wanted, "shared state " + name + " " + field)
        (Path(directory) / f"rank{rank}.json").write_text(json.dumps(dict(rank=rank,
            sp=sp, norms=norms, errors=evidence.errors, scales=evidence.scales)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("sp", [1, 2])
def test_shared_expert_gate_independent_objective(tmp_path, sp):
    mp.spawn(shared_worker, args=(str(tmp_path), sp), nprocs=4*sp, join=True)
    assert all((tmp_path / f"rank{rank}.json").exists() for rank in range(4*sp))
