"""Complete TP admission precedes runtime assignment and parameter mutation."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor import DTensor, Shard

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.tensor_parallel.expert import GroupedMLPParallel
from torchturbo.engine.tensor_parallel.parallelize import PrepareModuleMetadata
from torchturbo.engine.config import EngineConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer
from torchturbo.parallel import (
    FSDPParallelPlan, FSDPPlacementSpec, FSDPPlanSpec,
    TensorParallelPlanSpec, TensorParallelValidator,
)
from .test_expert_tensor_parallel import _experts
from .test_tensor_parallel import DenseDecoder, _native_plan


class RecordingValidator(TensorParallelValidator):
    def __init__(self, reject_call):
        self.reject_call = reject_call
        self.calls = []

    def validate(self, model, parallel_plan):
        self.calls.append((model.layers[0].up, (
            parallel_plan.sp_size, parallel_plan.ep_size,
            parallel_plan.tp_size, parallel_plan.etp_size,
        )))
        if len(self.calls) == self.reject_call:
            raise ValueError("Rejected model contract.")


def _backend(monkeypatch, *, routed=False):
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.config = EngineConfig()
    backend.config.moe.aux_loss_coeff = .02
    backend.config.moe.z_loss_coeff = .003
    backend.fsdp_config = backend.config.backend_config
    backend._parallel_context = None
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    group = object()
    mesh = SimpleNamespace(ndim=1, size=lambda: 2, get_group=lambda: group)
    context = SimpleNamespace(sp_size=2, ep_size=1, tp_size=2,
        etp_size=2 if routed else 1, tp_mesh=mesh, etp_mesh=mesh if routed else None,
        sp_mesh=mesh, expert_fsdp_mesh=mesh)

    def init_mesh(plan):
        backend._parallel_context = context

    monkeypatch.setattr(backend, "_init_mesh", init_mesh)

    def reject_mutation(*args, **kwargs):
        pytest.fail("TP mutation was reached before complete admission.")

    monkeypatch.setattr(
        "torchturbo.engine.tensor_parallel.parallelize.parallelize_module", reject_mutation
    )
    return backend


def _snapshot(model):
    return (
        {name: (parameter, parameter.detach().clone()) for name, parameter in model.named_parameters()},
        {name: (dict(module._forward_pre_hooks), dict(module._forward_hooks))
         for name, module in model.named_modules()},
        [(module, module.router.sp_group, module.router.routing_group,
          module.config.moe_aux_loss_coeff, module.config.moe_z_loss_coeff)
         for module in model.modules() if isinstance(module, MoELayer)],
    )


def _assert_unchanged(model, snapshot):
    parameters, hooks, runtime = snapshot
    assert dict(model.named_parameters()).keys() == parameters.keys()
    for name, (parameter, value) in parameters.items():
        assert model.get_parameter(name) is parameter
        assert not isinstance(parameter, DTensor)
        torch.testing.assert_close(parameter, value, rtol=0, atol=0)
    for name, (pre_hooks, post_hooks) in hooks.items():
        module = model.get_submodule(name)
        assert dict(module._forward_pre_hooks) == pre_hooks
        assert dict(module._forward_hooks) == post_hooks
    for module, sp, routing, aux, z in runtime:
        assert module.router.sp_group is sp
        assert module.router.routing_group is routing
        assert module.config.moe_aux_loss_coeff == aux
        assert module.config.moe_z_loss_coeff == z


def test_empty_tp_plan_rejected_before_mesh_or_module_replacement(monkeypatch):
    backend = _backend(monkeypatch)
    model = DenseDecoder()
    before = _snapshot(model)
    plan = _native_plan()
    plan.tensor_parallel_plan = TensorParallelPlanSpec()

    def reject_preparation(*args, **kwargs):
        pytest.fail("Empty TP plan reached mesh setup or module replacement.")

    monkeypatch.setattr(backend, "_init_mesh", reject_preparation)
    monkeypatch.setattr(backend, "_replace_modules", reject_preparation)

    with pytest.raises(ValueError, match="nonempty tensor_parallel_plan"):
        backend.prepare_model(model, plan)
    assert backend._parallel_context is None
    _assert_unchanged(model, before)


@pytest.mark.parametrize("reject_call", [1, 2])
def test_model_validator_rejects_before_mesh_or_tp_mutation(monkeypatch, reject_call):
    backend = _backend(monkeypatch)
    model = DenseDecoder()
    original = model.layers[0].up
    before = _snapshot(model)
    config = copy.deepcopy(vars(model.config))
    validator = RecordingValidator(reject_call)
    plan = _native_plan()
    plan.sp_size = 2
    plan.tensor_parallel_plan.validator = validator
    replacements = []

    def replace(current, current_plan):
        replacement = copy.deepcopy(current.layers[0].up)
        current.layers[0].up = replacement
        replacements.append(replacement)
        return current

    monkeypatch.setattr(backend, "_replace_modules", replace)
    with pytest.raises(ValueError, match="Rejected model contract"):
        backend.prepare_model(model, plan)
    assert validator.calls[0] == (original, (2, 1, 2, 1))
    assert model.layers[0].attn.num_heads == 8
    assert vars(model.config) == config
    if reject_call == 1:
        assert backend._parallel_context is None
        assert replacements == []
        _assert_unchanged(model, before)
    else:
        assert validator.calls[1] == (replacements[0], (2, 1, 2, 1))
        before[0]["layers.0.up.weight"] = (
            replacements[0].weight, before[0]["layers.0.up.weight"][1])
        _assert_unchanged(model, before)


@pytest.mark.parametrize("degrees", [
    (1, 1, 0, 1), (0, 1, 2, 1), (1, -1, 2, 1),
    (1, 1, 2.0, 1), (1, 1, 2, True), (1, 1, False, 1),
    (1, 1, 3, 1), (4, 1, 2, 1), (1, 4, 2, 2),
])
def test_invalid_degrees_rejected_before_model_arithmetic(monkeypatch, degrees):
    backend = _backend(monkeypatch)
    model = DenseDecoder()
    before = _snapshot(model)
    plan = _native_plan()
    plan.sp_size, plan.ep_size, plan.tp_size, plan.etp_size = degrees
    validator = RecordingValidator(1)
    plan.tensor_parallel_plan.validator = validator
    with pytest.raises(ValueError, match="positive integer|divide world size"):
        backend.prepare_model(model, plan)
    assert validator.calls == []
    assert backend._parallel_context is None
    _assert_unchanged(model, before)


def _routed_model():
    model = DenseDecoder()
    for name in ("first", "last"):
        # Admission needs the prepared shared layer inventory, not a dispatcher.
        layer = MoELayer.__new__(MoELayer)
        nn.Module.__init__(layer)
        layer.experts = _experts(num_experts=2)
        layer.config = layer.experts.config
        layer.config.moe_aux_loss_coeff = .5
        layer.config.moe_z_loss_coeff = .4
        layer.num_local_experts = 2
        layer.router = nn.Module()
        layer.router.sp_group = object()
        layer.router.routing_group = object()
        model.add_module(name, layer)
    return model


def _routed_plan():
    return FSDPParallelPlan(sp_size=2, tp_size=2, etp_size=2,
        plan={name + ".experts": FSDPPlanSpec(
            placement=FSDPPlacementSpec(Shard(0), "edp")) for name in ("first", "last")},
        tensor_parallel_plan=TensorParallelPlanSpec(validator=RecordingValidator(0), plan={
            "layers.0.up": ColwiseParallel(),
            "first.experts": GroupedMLPParallel(),
            "last.experts": GroupedMLPParallel(),
        }))


@pytest.mark.parametrize("broken", [
    "nongated", "te", "input_probs", "features", "wrong_module",
    "weight1_numel", "weight2_numel", "weight1_layout", "weight2_layout",
    "missing_style", "missing_unit", "stray_style", "dense_edp", "parameter_edp",
    "metadata", "prepared_metadata", "prepared_dropout", "vocabulary_head", "vocabulary_embed", "prepared_validator",
])
def test_prepared_rejection_preserves_all_targets_and_router(monkeypatch, broken):
    backend = _backend(monkeypatch, routed=True)
    model = _routed_model()
    plan = _routed_plan()
    expert = model.last.experts
    if broken == "nongated":
        expert.config.gated_linear_unit = False
    elif broken == "te":
        expert.use_te = True
    elif broken == "input_probs":
        expert.config.moe_apply_probs_on_input = True
    elif broken == "features":
        expert.config.moe_ffn_hidden_size = 3
    elif broken == "wrong_module":
        model.last.experts = nn.Linear(3, 4)
    elif broken.endswith("numel") or broken.endswith("layout"):
        name = broken.split("_")[0]
        parameter = expert.get_parameter(name)
        shape = (parameter.numel() - 1,) if broken.endswith("numel") else tuple(reversed(parameter.shape))
        expert.register_parameter(name, nn.Parameter(torch.zeros(shape)))
    elif broken == "missing_style":
        del plan.tensor_parallel_plan.plan["last.experts"]
    elif broken == "missing_unit":
        del plan.plan["last.experts"]
    elif broken == "stray_style":
        model.stray = _experts()
        plan.tensor_parallel_plan.plan["stray"] = GroupedMLPParallel()
    elif broken in ("dense_edp", "parameter_edp"):
        target = "norm" if broken == "dense_edp" else "norm.weight"
        plan.plan[target] = FSDPPlanSpec(placement=FSDPPlacementSpec(Shard(0), "edp"))
    elif broken == "metadata":
        model.layers[0].attn.num_key_value_heads = 3
        plan.tensor_parallel_plan.plan["layers.0.attn"] = PrepareModuleMetadata(
            divide=("num_heads", "num_key_value_heads"))
    elif broken.startswith("vocabulary"):
        target = "head" if broken.endswith("head") else "embed"
        plan.tensor_parallel_plan.plan[target] = RowwiseParallel() if target == "head" else ColwiseParallel()
    elif broken == "prepared_validator":
        plan.tensor_parallel_plan.validator = RecordingValidator(2)
    snapshot = _snapshot(model)
    replacement_calls = []

    def replace(current, current_plan):
        replacement_calls.append(current)
        if broken == "prepared_metadata":
            current.layers[0].attn.num_key_value_heads = 3
            plan.tensor_parallel_plan.plan["layers.0.attn"] = PrepareModuleMetadata(
                divide=("num_heads", "num_key_value_heads"))
        elif broken == "prepared_dropout":
            current.dropout = nn.Dropout(.1)
        return current

    monkeypatch.setattr(backend, "_replace_modules", replace)
    messages = dict(nongated="gated_linear_unit", te="use_te", input_probs="moe_apply_probs",
        features="divisible", wrong_module="shared GroupedMLP", missing_style="GroupedMLPParallel",
        missing_unit="edp FSDP", stray_style="shared MoELayer", dense_edp="reserved",
        parameter_edp="reserved", metadata="num_key_value_heads",
        prepared_metadata="num_key_value_heads", prepared_dropout="dropout", prepared_validator="Rejected model")
    message = "weight" if broken.startswith("weight") else "vocabulary" if broken.startswith("vocabulary") else messages[broken]
    with pytest.raises(ValueError, match=message):
        backend.prepare_model(model, plan)
    _assert_unchanged(model, snapshot)
    assert model.layers[0].attn.num_heads == 8
    if broken != "metadata":
        assert replacement_calls == [model]


@pytest.mark.parametrize("field", ["moe_ffn_hidden_size", "hidden_size", "num_local_experts"])
def test_late_expert_float_metadata_rejects_before_router_or_tp_mutation(monkeypatch, field):
    backend = _backend(monkeypatch, routed=True)
    model = _routed_model()
    plan = _routed_plan()
    expert = model.last.experts
    owner = expert if field == "num_local_experts" else expert.config
    setattr(owner, field, float(vars(owner)[field]))
    before = _snapshot(model)
    monkeypatch.setattr(backend, "_replace_modules", lambda current, plan: current)
    with pytest.raises(ValueError, match=field + ".*integer"):
        backend.prepare_model(model, plan)
    _assert_unchanged(model, before)


@pytest.mark.parametrize("stage", ["original", "prepared"])
@pytest.mark.parametrize("field", ["num_k_heads", "num_v_heads", "key_dim", "value_dim", "conv_dim"])
def test_late_gdn_float_metadata_rejects_before_router_or_tp_mutation(monkeypatch, stage, field):
    from tests.models.qwen3_5_moe.tp_helpers import text_model
    from torchturbo.models.qwen3_5_moe_text.parallel_plan import get_parallel_plan

    backend = _backend(monkeypatch, routed=True)
    model = text_model(layer_types=["full_attention", "linear_attention"])
    model.earlier = nn.Linear(16, 16, bias=False)
    plan = get_parallel_plan()
    plan.sp_size = plan.tp_size = plan.etp_size = 2
    plan.tensor_parallel_plan.plan = {"earlier": ColwiseParallel(), **plan.tensor_parallel_plan.plan}
    gdn = model.model.layers[1].linear_attn
    initial = vars(gdn)[field]
    if stage == "original":
        setattr(gdn, field, float(initial))
    original = _snapshot(model)
    prepared_snapshots = []

    def replace(current, current_plan):
        assert stage == "prepared", "Original invalid metadata reached replacement."
        for layer in current.model.layers:
            original_mlp = layer.mlp
            prepared = MoELayer.__new__(MoELayer)
            nn.Module.__init__(prepared)
            prepared.experts = _experts(num_experts=4, hidden_size=16, moe_ffn_hidden_size=12)
            prepared.config = prepared.experts.config
            prepared.num_local_experts = 4
            prepared.shared_expert = original_mlp.shared_expert
            prepared.shared_expert_gate = original_mlp.shared_expert_gate
            prepared.router = nn.Module()
            prepared.router.sp_group = object()
            prepared.router.routing_group = object()
            layer.mlp = prepared
        setattr(gdn, field, float(initial))
        prepared_snapshots.append(_snapshot(current))
        return current

    monkeypatch.setattr(backend, "_replace_modules", replace)
    with pytest.raises(ValueError, match=field + ".*positive.*integer"):
        backend.prepare_model(model, plan)
    _assert_unchanged(model, original if stage == "original" else prepared_snapshots[0])
    assert vars(gdn)[field] == initial
    assert type(vars(gdn)[field]) is float
    assert not prepared_snapshots if stage == "original" else len(prepared_snapshots) == 1
    if stage == "original":
        assert backend._parallel_context is None
