"""Real shared-MoE fixtures for logical checkpoint acceptance."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.distributed.tensor import DTensor
from transformers import PretrainedConfig

from torchturbo.engine.checkpoint.coordinator import CheckpointCoordinator
from torchturbo.engine.optimizers import OptimizerClassPlan
from torchturbo.models.weight_converter import BaseWeightConverter, FusedExpertWeightConverter
from tests.engine.backends.fsdp2.test_etp_training import SharedMoETrainingModel, _backend, _full, _plan


class CheckpointMoE(SharedMoETrainingModel):
    def __init__(self, context, num_local_experts, *, fused=False):
        super().__init__(context.ep_mesh.get_group(),
                         num_moe_experts=num_local_experts * context.ep_size)
        self.config = PretrainedConfig(hidden_size=3, moe_intermediate_size=4)
        self.config.model_type = "shared_checkpoint_moe"
        self.all_tied_weights_keys = {"head_alias.weight": "head.weight"}
        self.head_alias = self.head
        self.fused = fused

    def tie_weights(self):
        self.head_alias = self.head

    def get_weight_converter(self, parallel_context, model_path):
        converter = FusedExpertWeightConverter if self.fused else BaseWeightConverter
        return converter(self, parallel_context, model_path)


def expert_values(num_experts):
    """Encode expert, gate/up, hidden and feature coordinates independently."""
    expert = torch.arange(num_experts)[:, None, None, None] * 1000
    hidden = torch.arange(3)[None, :, None, None] * 100
    projection = torch.arange(2)[None, None, :, None] * 10
    feature = torch.arange(4)[None, None, None, :]
    w1 = (expert + hidden + projection + feature).float() / 10000
    w2 = (torch.arange(num_experts)[:, None, None] * 1000
          + torch.arange(4)[None, :, None] * 10
          + torch.arange(3)[None, None, :]).float() / 10000
    return w1, w2


def hf_values(model, num_experts, *, fused=False):
    values = {}
    for name, parameter in model.named_parameters():
        if ".experts." not in name:
            hf_name = name.replace("moe.router.", "moe.gate.")
            values[hf_name] = parameter.detach().cpu().clone()
    w1, w2 = expert_values(num_experts)
    if fused:
        values["moe.experts.gate_up_proj"] = w1.flatten(2).transpose(1, 2).contiguous()
        values["moe.experts.down_proj"] = w2.transpose(1, 2).contiguous()
    else:
        for expert in range(num_experts):
            for index, name in enumerate(("gate_proj", "up_proj")):
                values[f"moe.experts.{expert}.{name}.weight"] = w1[expert, :, index].t().contiguous()
            values[f"moe.experts.{expert}.down_proj.weight"] = w2[expert].t().contiguous()
    return values


def write_source(path, values):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for index, (name, value) in enumerate(values.items()):
        filename = f"source-{index}.safetensors"
        save_file({name: value.contiguous()}, str(path / filename))
        weight_map[name] = filename
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(value.numel() * value.element_size() for value in values.values())},
        "weight_map": weight_map,
    }))
    (path / "config.json").write_text('{"model_type": "shared_checkpoint_moe"}')
    return weight_map


def read_hf(path):
    path = Path(path)
    index = json.loads((path / "model.safetensors.index.json").read_text())
    values = {}
    for filename in set(index["weight_map"].values()):
        values.update(load_file(str(path / filename)))
    assert values.keys() == index["weight_map"].keys()
    return values


def prepare_checkpoint_model(context, num_local_experts, *, meta=False, fused=False, dtype=torch.float32):
    with torch.device("meta" if meta else "cpu"):
        model = CheckpointMoE(context, num_local_experts, fused=fused)
        model.to(dtype)
    if not meta:
        w1, w2 = expert_values(num_local_experts * context.ep_size)
        start = context.ep_mesh.get_local_rank() * num_local_experts
        with torch.no_grad():
            model.moe.experts.weight1.copy_(w1[start:start + num_local_experts].flatten(2).flatten(0, 1))
            model.moe.experts.weight2.copy_(w2[start:start + num_local_experts].flatten(0, 1))
    backend = _backend(context)
    backend.config.parallel_plan = _plan(context)
    backend.config.checkpoint.hf_export = "final"
    if context.etp_size == 1:
        backend.config.parallel_plan.tp_size = 1
        backend.config.parallel_plan.etp_size = 1
        backend.config.parallel_plan.tensor_parallel_plan.plan.clear()
    backend.prepare_model(model, backend.config.parallel_plan)
    manager = backend.create_checkpoint_manager()
    coordinator = CheckpointCoordinator(backend.config, manager,
                                        parallel_context=backend.get_parallel_context())
    return model, backend, manager, coordinator


def training_step(model, optimizer, context, step):
    optimizer.zero_grad(set_to_none=True)
    # Every logical DP replica supplies its own batch; TP lanes share it.
    replica = context.dense_fsdp_mesh.get_local_rank()
    inputs = torch.arange(18., dtype=model.head.weight.dtype).view(2, 3, 3) / 19 + .1 * step + .07 * replica
    output = model(inputs)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()


def prepare_checkpoint_optimizer(model, backend, *, lr=.003):
    return backend.prepare_optimizer(OptimizerClassPlan(
        model, torch.optim.AdamW, {"lr": lr, "weight_decay": .01, "foreach": False}))


def assert_exact_model_and_optimizer(left, left_optimizer, right, right_optimizer):
    from tests.engine.backends.fsdp2.test_tp_checkpoint import _assert_state_equal
    _assert_state_equal(left.state_dict(), right.state_dict())
    _assert_state_equal(left_optimizer.state_dict(), right_optimizer.state_dict())
    for name, parameter in left.named_parameters():
        other = right.get_parameter(name)
        assert_tensor_bytes_equal(parameter, other)
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert_tensor_bytes_equal(left_optimizer.state[parameter][field],
                                      right_optimizer.state[other][field])


def assert_tensor_bytes_equal(left, right):
    pairs = [(_full(left), _full(right))]
    if isinstance(left, DTensor):
        pairs.append((left.to_local(), right.to_local()))
    for actual, expected in pairs:
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        actual_bytes = actual.detach().contiguous().reshape(-1).view(torch.uint8)
        expected_bytes = expected.detach().contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(actual_bytes, expected_bytes)
