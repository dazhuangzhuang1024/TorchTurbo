"""Tests for FSDP2 DCP model and optimizer state."""

import os

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch
from torch import nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from transformers import PretrainedConfig, PreTrainedModel

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.checkpoint.dcp import DcpCheckpoint
from torchturbo.engine.backends.fsdp2.config import FSDPConfig
from torchturbo.engine.optimizers import OptimizerClassPlan
from torchturbo.engine.backends.fsdp2.parallel_context import ParameterOwnership
from torchturbo.runtime.cpu import CpuRuntime


def _checkpoint(model: nn.Module) -> DcpCheckpoint:
    # This model has no expert units. DCP supports single-process CPU save/load
    # without a process group.
    ownership = ParameterOwnership()
    ownership.register(model, set())
    return DcpCheckpoint(ep_rank=0, ep_size=1, ownership=ownership)


class FrozenBaseModel(nn.Module):
    """A frozen base layer next to a trainable layer and a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 3)
        self.adapter = nn.Linear(3, 3)
        self.register_buffer("bias_state", torch.zeros(3))
        self.base.requires_grad_(False)


class SmallerModel(nn.Module):
    """A model that consumes a subset of a larger checkpoint's state."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 3)


class LargerModel(SmallerModel):
    def __init__(self) -> None:
        super().__init__()
        self.checkpoint_only = nn.Linear(3, 3)


def _saved_model_keys(model_dir: str) -> set[str]:
    metadata = FileSystemReader(model_dir).read_metadata()
    return set(metadata.state_dict_metadata)


def test_expert_ownership_preserves_legacy_canonical_keys_and_aliases(tmp_path):
    model = nn.Module()
    model.dense = nn.Linear(2, 2, bias=False)
    model.expert = nn.Linear(2, 2, bias=False)
    model.alias = model.expert
    ownership = ParameterOwnership()
    ownership.register(model, {model.expert.weight})
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, ownership=ownership)
    model_dir = str(tmp_path / "model")
    expected = {name: value.clone() for name, value in model.state_dict().items()}

    checkpoint.save_model(model, model_dir)

    assert _saved_model_keys(model_dir) == {
        "model.dense.weight", "model.expert.weight::ep0", "model.alias.weight",
    }
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint.load_model(model, model_dir)
    torch.testing.assert_close(model.state_dict(), expected)


def test_load_adamw_checkpoint_without_parameter_names(tmp_path):
    torch.manual_seed(7)
    reference_model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    reference_optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in reference_model.parameters() if p.ndim > 1],
                "weight_decay": 0.1,
            },
            {
                "params": [p for p in reference_model.parameters() if p.ndim == 1],
                "weight_decay": 0.0,
            },
        ],
        lr=0.002,
        weight_decay=0.1,
    )
    for step in range(3):
        for index, param in enumerate(reference_model.parameters()):
            param.grad = torch.full_like(param, (step + 1) * (index + 1) / 10)
        reference_optimizer.step()
        reference_optimizer.zero_grad()

    checkpoint = _checkpoint(reference_model)
    model_dir = str(tmp_path / "model")
    optimizer_dir = str(tmp_path / "optimizer")
    checkpoint.save_model(reference_model, model_dir)
    checkpoint.save_optimizer(reference_optimizer, reference_model, optimizer_dir)

    resumed_model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    checkpoint.ownership.register(resumed_model, set())
    plan = OptimizerClassPlan(
        resumed_model,
        torch.optim.AdamW,
        {"lr": 0.002, "weight_decay": 0.1},
    )
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend._parallel_context = None
    backend._device_runtime = CpuRuntime()
    backend.fsdp_config = FSDPConfig()
    backend._mesh = {"edp": object()}
    resumed_optimizer = backend.prepare_optimizer(plan)

    checkpoint.load_model(resumed_model, model_dir)
    checkpoint.load_optimizer(resumed_optimizer, resumed_model, optimizer_dir)
    torch.testing.assert_close(
        get_optimizer_state_dict(resumed_model, resumed_optimizer)["state"],
        get_optimizer_state_dict(reference_model, reference_optimizer)["state"],
        rtol=0,
        atol=0,
    )

    for model, optimizer in (
        (reference_model, reference_optimizer),
        (resumed_model, resumed_optimizer),
    ):
        for index, param in enumerate(model.parameters()):
            param.grad = torch.full_like(param, -(index + 1) / 5)
        optimizer.step()

    torch.testing.assert_close(
        resumed_model.state_dict(), reference_model.state_dict(), rtol=0, atol=0
    )
    resumed_state = get_optimizer_state_dict(resumed_model, resumed_optimizer)
    reference_state = get_optimizer_state_dict(reference_model, reference_optimizer)
    assert resumed_state["param_groups"] == reference_state["param_groups"]
    torch.testing.assert_close(
        resumed_state["state"], reference_state["state"], rtol=0, atol=0
    )


def test_model_state_stores_frozen_parameters_and_buffers(tmp_path):
    torch.manual_seed(0)
    trained = FrozenBaseModel()
    trained.bias_state.fill_(3.0)
    model_dir = str(tmp_path / "model")
    _checkpoint(trained).save_model(trained, model_dir)

    assert _saved_model_keys(model_dir) == {
        "model.base.weight",
        "model.base.bias",
        "model.adapter.weight",
        "model.adapter.bias",
        "model.bias_state",
    }

    torch.manual_seed(1)
    resumed = FrozenBaseModel()
    _checkpoint(resumed).load_model(resumed, model_dir)

    torch.testing.assert_close(
        resumed.state_dict(), trained.state_dict(), rtol=0, atol=0
    )


def test_load_restores_stored_entries_that_are_frozen_now(tmp_path):
    torch.manual_seed(0)
    trained = FrozenBaseModel()
    trained.base.requires_grad_(True)
    model_dir = str(tmp_path / "model")
    _checkpoint(trained).save_model(trained, model_dir)

    torch.manual_seed(1)
    resumed = FrozenBaseModel()
    _checkpoint(resumed).load_model(resumed, model_dir)

    torch.testing.assert_close(
        resumed.state_dict(), trained.state_dict(), rtol=0, atol=0
    )


class TiedModel(PreTrainedModel):
    """An embedding shared with an output projection."""

    def __init__(self) -> None:
        super().__init__(PretrainedConfig())
        self.embed = nn.Embedding(4, 3)
        self.head = nn.Linear(3, 4, bias=False)
        self.tie_weights()

    def tie_weights(self) -> None:
        self.head.weight = self.embed.weight


def test_load_preserves_tied_parameter_identity(tmp_path):
    torch.manual_seed(0)
    trained = TiedModel()
    model_dir = str(tmp_path / "model")
    _checkpoint(trained).save_model(trained, model_dir)

    resumed = TiedModel()
    original_parameter = resumed.embed.weight

    _checkpoint(resumed).load_model(resumed, model_dir)

    assert resumed.embed.weight is original_parameter
    assert resumed.head.weight is original_parameter
    torch.testing.assert_close(
        resumed.embed.weight, trained.embed.weight, rtol=0, atol=0
    )


class CloningStateDictModel(nn.Module):
    """A module whose state-dict hook detaches values from live parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3))
        self.register_state_dict_post_hook(self._clone_state_dict)

    @staticmethod
    def _clone_state_dict(module, state_dict, prefix, local_metadata) -> None:
        for name, value in state_dict.items():
            state_dict[name] = value.clone()


def test_load_applies_populated_state_through_model_load_state_dict(tmp_path):
    saved = CloningStateDictModel()
    saved.weight.data.fill_(1.0)
    model_dir = str(tmp_path / "model")
    _checkpoint(saved).save_model(saved, model_dir)

    resumed = CloningStateDictModel()
    resumed.weight.data.fill_(9.0)
    _checkpoint(resumed).load_model(resumed, model_dir)

    torch.testing.assert_close(resumed.weight, saved.weight, rtol=0, atol=0)


def test_load_rejects_missing_entries(tmp_path):
    model_dir = str(tmp_path / "model")
    saved = SmallerModel()
    resumed = LargerModel()
    _checkpoint(saved).save_model(saved, model_dir)
    with pytest.raises(RuntimeError, match=r"missing entries.*checkpoint_only"):
        _checkpoint(resumed).load_model(resumed, model_dir)


def test_load_ignores_checkpoint_only_entries(tmp_path):
    model_dir = str(tmp_path / "model")
    saved = LargerModel()
    _checkpoint(saved).save_model(saved, model_dir)
    resumed = SmallerModel()
    _checkpoint(resumed).load_model(resumed, model_dir)
    torch.testing.assert_close(resumed.shared.state_dict(), saved.shared.state_dict())


def test_ep_size_validation_rejects_a_different_size():
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, ownership=ParameterOwnership())
    saved_keys = {
        "model.experts.weight::ep0",
        "model.experts.weight::ep1",
    }

    with pytest.raises(ValueError, match=r"saved experts\.weight on EP ranks \[0, 1\]"):
        checkpoint._validate_ep_size(
            saved_keys,
            frozenset({"experts.weight"}),
            "/checkpoint/model_distcp",
        )


def test_ep_size_validation_ignores_checkpoint_only_entries():
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, ownership=ParameterOwnership())
    saved_keys = {
        "model.removed_experts.weight::ep0",
        "model.removed_experts.weight::ep1",
    }

    checkpoint._validate_ep_size(
        saved_keys,
        frozenset({"current_experts.weight"}),
        "/checkpoint/model_distcp",
    )


def test_ep_size_validation_rejects_a_missing_rank():
    checkpoint = DcpCheckpoint(ep_rank=1, ep_size=2, ownership=ParameterOwnership())
    saved_keys = {"model.experts.weight::ep1"}

    with pytest.raises(ValueError, match=r"saved experts\.weight on EP ranks \[1\]"):
        checkpoint._validate_ep_size(
            saved_keys,
            frozenset({"experts.weight"}),
            "/checkpoint/model_distcp",
        )


def test_trained_then_frozen_weights_survive_subsequent_checkpoints(tmp_path):
    trained = FrozenBaseModel()
    trained.base.weight.data.fill_(3)
    checkpoint = _checkpoint(trained)
    checkpoint.save_model(trained, str(tmp_path / "first"))
    resumed = FrozenBaseModel()
    resumed.base.weight.data.fill_(1)
    checkpoint.ownership.register(resumed, set())
    checkpoint.load_model(resumed, str(tmp_path / "first"))
    checkpoint.save_model(resumed, str(tmp_path / "second"))
    restored = FrozenBaseModel()
    checkpoint.ownership.register(restored, set())
    checkpoint.load_model(restored, str(tmp_path / "second"))
    torch.testing.assert_close(restored.state_dict(), trained.state_dict())


class ExtraStateModel(nn.Module):
    def __init__(self, version):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.extra = {"version": version, "nested": {"value": torch.tensor([version])}}

    def get_extra_state(self):
        return self.extra

    def set_extra_state(self, state):
        self.extra = state


def test_nested_extra_state_round_trip(tmp_path):
    model = ExtraStateModel(7)
    _checkpoint(model).save_model(model, str(tmp_path))
    resumed = ExtraStateModel(0)
    _checkpoint(resumed).load_model(resumed, str(tmp_path))
    torch.testing.assert_close(resumed.state_dict(), model.state_dict())


def test_dcp_rejects_missing_base_weights_after_removing_lora_wrappers(tmp_path):
    from peft import LoraConfig, get_peft_model

    def build(targets):
        base = nn.Module()
        base.a = nn.Linear(2, 2, bias=False)
        base.b = nn.Linear(2, 2, bias=False)
        return get_peft_model(base, LoraConfig(r=1, target_modules=targets))

    model = build(["a", "b"])
    _checkpoint(model).save_model(model, str(tmp_path))
    resumed = build(["a"])
    with pytest.raises(RuntimeError, match=r"missing entries.*b.weight"):
        _checkpoint(resumed).load_model(resumed, str(tmp_path))


@pytest.mark.parametrize("component", ["model", "optimizer"])
def test_tp_dcp_shape_preflight_precedes_tensor_and_moment_mutation(tmp_path, component):
    saved = nn.Linear(3, 4, bias=False)
    ownership = ParameterOwnership()
    ownership.register(saved, set())
    checkpoint = DcpCheckpoint(0, 1, ownership, stamp_all_aliases=True)
    optimizer = torch.optim.AdamW(saved.parameters())
    saved(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    current = nn.Linear(6, 2, bias=False)
    ownership.register(current, set())
    current_optimizer = torch.optim.AdamW(current.parameters())
    before = current.weight.detach().clone()
    if component == "model":
        checkpoint.save_model(saved, str(tmp_path))
        load = lambda: checkpoint.load_model(current, str(tmp_path))
    else:
        checkpoint.save_optimizer(optimizer, saved, str(tmp_path))
        load = lambda: checkpoint.load_optimizer(current_optimizer, current, str(tmp_path))
    with pytest.raises(ValueError, match="shape"):
        load()
    torch.testing.assert_close(current.weight, before, rtol=0, atol=0)
    assert not current_optimizer.state


def test_tp_dcp_checks_second_adamw_moment_before_allocating_state(tmp_path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
    model = nn.Linear(3, 4, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    state = get_optimizer_state_dict(model, optimizer)
    state["state"]["weight"]["exp_avg_sq"] = torch.ones(2, 6)
    dcp.save({"optimizer": state}, checkpoint_id=str(tmp_path))
    ownership = ParameterOwnership()
    ownership.register(model, set())
    checkpoint = DcpCheckpoint(0, 1, ownership, stamp_all_aliases=True)
    fresh = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="exp_avg_sq.*shape"):
        checkpoint.load_optimizer(fresh, model, str(tmp_path))
    assert not fresh.state


def test_tp_dcp_checks_persistent_buffer_shape_before_parameter_mutation(tmp_path):
    class Buffered(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.register_buffer("history", torch.ones(size))
    saved, current = Buffered(3), Buffered(2)
    ownership = ParameterOwnership()
    ownership.register(saved, set())
    ownership.register(current, set())
    checkpoint = DcpCheckpoint(0, 1, ownership, stamp_all_aliases=True)
    checkpoint.save_model(saved, str(tmp_path))
    with torch.no_grad():
        current.weight.zero_()
    with pytest.raises(ValueError, match="history.*shape"):
        checkpoint.load_model(current, str(tmp_path))
    assert current.weight.tolist() == [0, 0]


@pytest.mark.parametrize("missing", ["exp_avg", "exp_avg_sq", "step", "max_exp_avg_sq", "params", "lr"])
@pytest.mark.parametrize("initialized", [False, True], ids=["fresh", "existing"])
def test_tp_incomplete_adamw_checkpoint_rejects_before_state_mutation(tmp_path, monkeypatch, missing, initialized):
    import copy
    import torch.distributed.checkpoint as dcp
    saved = nn.Linear(3, 4, bias=False)
    saved_optimizer = torch.optim.AdamW(saved.parameters(), amsgrad=True)
    saved(torch.ones(2, 3)).sum().backward()
    saved_optimizer.step()
    state = get_optimizer_state_dict(saved, saved_optimizer)
    if missing in ("params", "lr"):
        del state["param_groups"][0][missing]
    else:
        del state["state"]["weight"][missing]
    dcp.save({"optimizer": state}, checkpoint_id=str(tmp_path))
    current = nn.Linear(3, 4, bias=False)
    optimizer = torch.optim.AdamW(current.parameters(), amsgrad=True)
    if initialized:
        current(torch.full((2, 3), 2.0)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    before = copy.deepcopy(optimizer.state_dict())
    weight = current.weight.detach().clone()
    ownership = ParameterOwnership()
    ownership.register(current, set())
    checkpoint = DcpCheckpoint(0, 1, ownership, stamp_all_aliases=True)
    monkeypatch.setattr(dcp, "load", lambda *args, **kwargs: pytest.fail("DCP load entered before required-key preflight"))
    with pytest.raises(ValueError, match=f"missing.*{missing}"):
        checkpoint.load_optimizer(optimizer, current, str(tmp_path))
    torch.testing.assert_close(optimizer.state_dict(), before, rtol=0, atol=0)
    torch.testing.assert_close(current.weight, weight, rtol=0, atol=0)
    assert current.weight.grad is None


def test_tp_adamw_preflight_preserves_excluded_and_frozen_parameters(tmp_path):
    model = nn.Module()
    model.trained = nn.Linear(3, 4, bias=False)
    model.frozen = nn.Linear(3, 4, bias=False)
    model.excluded = nn.Linear(3, 4, bias=False)
    model.frozen.requires_grad_(False)
    optimizer = torch.optim.AdamW([model.trained.weight, model.frozen.weight])
    model.trained(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    ownership = ParameterOwnership()
    ownership.register(model, set())
    checkpoint = DcpCheckpoint(0, 1, ownership, stamp_all_aliases=True)
    checkpoint.save_optimizer(optimizer, model, str(tmp_path))
    current = nn.Module()
    current.trained = nn.Linear(3, 4, bias=False)
    current.frozen = nn.Linear(3, 4, bias=False)
    current.frozen.requires_grad_(False)
    current.excluded = nn.Linear(6, 2, bias=False)
    ownership.register(current, set())
    restored = torch.optim.AdamW([current.trained.weight, current.frozen.weight])
    checkpoint.load_optimizer(restored, current, str(tmp_path))
    assert set(restored.state) == {current.trained.weight}
    torch.testing.assert_close(restored.state[current.trained.weight], optimizer.state[model.trained.weight], rtol=0, atol=0)


class SparseAdamModel(nn.Module):
    """Two trainable leaves whose participation changes on the next update."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(3, 4, bias=False)
        self.b = nn.Linear(3, 4, bias=False)
        self.register_buffer("bias_state", torch.tensor([.2, -.1]))


def sparse_optimizer(model, *, reverse=False):
    groups = [dict(params=[model.a.weight], lr=.003),
              dict(params=[model.b.weight], lr=.009)]
    return torch.optim.AdamW(groups[::-1] if reverse else groups, weight_decay=.1,
                             foreach=False, fused=False)


def sparse_step(model, optimizer, *, leaves=("a",), update=True):
    optimizer.zero_grad(set_to_none=True)
    for index, name in enumerate(leaves):
        model.get_submodule(name)(torch.full((2, 3), .2 + index * .3)).square().sum().backward()
    if update:
        optimizer.step()


def sparse_checkpoint(model):
    ownership = ParameterOwnership()
    ownership.register(model, set())
    return DcpCheckpoint(0, 1, ownership, stamp_all_aliases=True)


@pytest.mark.parametrize("superset", [False, True], ids=["fresh", "superset"])
def test_tp_sparse_adam_restore_and_first_use_continuation(tmp_path, superset):
    import copy
    from tests.engine.backends.fsdp2.test_tp_checkpoint import _assert_state_equal
    saved = SparseAdamModel()
    optimizer = sparse_optimizer(saved)
    sparse_step(saved, optimizer)
    assert set(optimizer.state) == {saved.a.weight}
    checkpoint = sparse_checkpoint(saved)
    checkpoint.save_model(saved, str(tmp_path / "model"))
    before_save = copy.deepcopy(optimizer.state_dict())
    checkpoint.save_optimizer(optimizer, saved, str(tmp_path / "optimizer"))
    _assert_state_equal(optimizer.state_dict(), before_save)
    current = SparseAdamModel()
    current_optimizer = sparse_optimizer(current)
    if superset:
        sparse_step(current, current_optimizer, leaves=("a", "b"))
    checkpoint.ownership.register(current, set())
    identities = {name: id(p) for name, p in current.named_parameters()}
    checkpoint.load_model(current, str(tmp_path / "model"))
    checkpoint.load_optimizer(current_optimizer, current, str(tmp_path / "optimizer"))
    _assert_state_equal(saved.state_dict(), current.state_dict())
    _assert_state_equal(optimizer.state_dict(), current_optimizer.state_dict())
    assert identities == {name: id(p) for name, p in current.named_parameters()}
    assert set(current_optimizer.state) == {current.a.weight}
    for model, opt in ((saved, optimizer), (current, current_optimizer)):
        sparse_step(model, opt, leaves=("a", "b"))
        assert opt.state[model.b.weight]["step"] == 1
        assert opt.state[model.a.weight]["step"] == 2
    _assert_state_equal(saved.state_dict(), current.state_dict())
    _assert_state_equal(optimizer.state_dict(), current_optimizer.state_dict())


def test_tp_slot_free_pending_gradient_checkpoint_restores_empty_state(tmp_path):
    import copy
    from tests.engine.backends.fsdp2.test_tp_checkpoint import _assert_state_equal
    saved = SparseAdamModel()
    optimizer = sparse_optimizer(saved)
    sparse_step(saved, optimizer, leaves=("a", "b"), update=False)
    assert not optimizer.state
    gradients = {name: p.grad.clone() for name, p in saved.named_parameters()}
    state_before = copy.deepcopy(saved.state_dict())
    groups_before = copy.deepcopy(optimizer.state_dict())
    checkpoint = sparse_checkpoint(saved)
    checkpoint.save_model(saved, str(tmp_path / "model"))
    checkpoint.save_optimizer(optimizer, saved, str(tmp_path / "optimizer"))
    _assert_state_equal(saved.state_dict(), state_before)
    _assert_state_equal(optimizer.state_dict(), groups_before)
    for name, p in saved.named_parameters():
        torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)
    current = SparseAdamModel()
    current_optimizer = sparse_optimizer(current)
    checkpoint.ownership.register(current, set())
    identities = {name: id(p) for name, p in current.named_parameters()}
    checkpoint.load_model(current, str(tmp_path / "model"))
    checkpoint.load_optimizer(current_optimizer, current, str(tmp_path / "optimizer"))
    assert not current_optimizer.state
    _assert_state_equal(current_optimizer.state_dict(), groups_before)
    assert identities == {name: id(p) for name, p in current.named_parameters()}
    for model, opt in ((saved, optimizer), (current, current_optimizer)):
        sparse_step(model, opt, leaves=("a", "b"))
        assert all(slot["step"] == 1 for slot in opt.state.values())
    _assert_state_equal(saved.state_dict(), current.state_dict())
    _assert_state_equal(optimizer.state_dict(), current_optimizer.state_dict())


@pytest.mark.parametrize("target", ["uncovered", "pending"])
def test_tp_saved_slot_without_native_skeleton_rejects_before_mutation(tmp_path, monkeypatch, target):
    import copy
    import torch.distributed.checkpoint as dcp
    from tests.engine.backends.fsdp2.test_tp_checkpoint import _assert_state_equal
    saved, current = SparseAdamModel(), SparseAdamModel()
    optimizer, current_optimizer = sparse_optimizer(saved), sparse_optimizer(current)
    sparse_step(saved, optimizer)
    checkpoint = sparse_checkpoint(saved)
    checkpoint.save_optimizer(optimizer, saved, str(tmp_path))
    sparse_step(current, current_optimizer, leaves=("b",), update=target == "uncovered")
    checkpoint.ownership.register(current, set())
    before = copy.deepcopy(current_optimizer.state_dict())
    model_before = copy.deepcopy(current.state_dict())
    gradients = {name: None if p.grad is None else p.grad.clone() for name, p in current.named_parameters()}
    monkeypatch.setattr(dcp, "load", lambda *args, **kwargs: pytest.fail("DCP entered before skeleton coverage"))
    with pytest.raises(ValueError, match="Native.*skeleton.*a.weight"):
        checkpoint.load_optimizer(current_optimizer, current, str(tmp_path))
    _assert_state_equal(current_optimizer.state_dict(), before)
    _assert_state_equal(current.state_dict(), model_before)
    for name, p in current.named_parameters():
        if gradients[name] is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)


def test_tp_saved_trainable_group_member_deletion_rejects_before_setter(tmp_path, monkeypatch):
    import torch.distributed.checkpoint as dcp
    from torchturbo.engine.backends.fsdp2.checkpoint import dcp as checkpoint_module
    model = SparseAdamModel()
    optimizer = sparse_optimizer(model)
    sparse_step(model, optimizer)
    state = get_optimizer_state_dict(model, optimizer)
    state["param_groups"][1]["params"].clear()
    dcp.save({"optimizer": state}, checkpoint_id=str(tmp_path))
    current = SparseAdamModel()
    current_optimizer = sparse_optimizer(current)
    checkpoint = sparse_checkpoint(current)
    monkeypatch.setattr(checkpoint_module, "set_optimizer_state_dict",
                        lambda *args, **kwargs: pytest.fail("setter entered before loaded membership check"))
    with pytest.raises(ValueError, match="Saved.*group.*b.weight"):
        checkpoint.load_optimizer(current_optimizer, current, str(tmp_path))
    # Membership is loaded by DCP after native transient initialization.
    assert set(current_optimizer.state) == {current.a.weight, current.b.weight}


def test_tp_empty_state_reordered_distinct_lr_groups_reject_before_setter(tmp_path, monkeypatch):
    from torchturbo.engine.backends.fsdp2.checkpoint import dcp as checkpoint_module
    model = SparseAdamModel()
    optimizer = sparse_optimizer(model)
    sparse_step(model, optimizer, leaves=("a", "b"), update=False)
    checkpoint = sparse_checkpoint(model)
    checkpoint.save_optimizer(optimizer, model, str(tmp_path))
    current = SparseAdamModel()
    current_optimizer = sparse_optimizer(current, reverse=True)
    checkpoint.ownership.register(current, set())
    monkeypatch.setattr(checkpoint_module, "set_optimizer_state_dict",
                        lambda *args, **kwargs: pytest.fail("setter entered before empty-state group check"))
    with pytest.raises(ValueError, match="Empty optimizer state group 0.*b.weight"):
        checkpoint.load_optimizer(current_optimizer, current, str(tmp_path))
    assert [group["lr"] for group in current_optimizer.param_groups] == [.009, .003]


def test_tp_frozen_existing_target_slot_still_requires_saved_state(tmp_path, monkeypatch):
    import copy
    import torch.distributed.checkpoint as dcp
    from tests.engine.backends.fsdp2.test_tp_checkpoint import _assert_state_equal
    saved, current = SparseAdamModel(), SparseAdamModel()
    optimizer, current_optimizer = sparse_optimizer(saved), sparse_optimizer(current)
    sparse_step(saved, optimizer)
    checkpoint = sparse_checkpoint(saved)
    checkpoint.save_optimizer(optimizer, saved, str(tmp_path))
    sparse_step(current, current_optimizer, leaves=("a", "b"))
    current.b.requires_grad_(False)
    checkpoint.ownership.register(current, set())
    before = copy.deepcopy(current_optimizer.state_dict())
    monkeypatch.setattr(dcp, "load", lambda *args, **kwargs: pytest.fail("DCP entered before frozen slot check"))
    with pytest.raises(ValueError, match="missing.*b.weight.step"):
        checkpoint.load_optimizer(current_optimizer, current, str(tmp_path))
    _assert_state_equal(current_optimizer.state_dict(), before)


def test_tp_preflight_admits_trainable_ep_state_absent_under_every_alias(tmp_path):
    model = nn.Module()
    model.expert = nn.Linear(3, 4, bias=False)
    ownership = ParameterOwnership()
    ownership.register(model, {model.expert.weight})
    checkpoint = DcpCheckpoint(0, 2, ownership, stamp_all_aliases=True)
    optimizer = torch.optim.AdamW(model.parameters())
    # Pending gradients preserve native lazy state before the first update.
    model.expert.weight.grad = torch.ones_like(model.expert.weight)
    checkpoint.save_optimizer(optimizer, model, str(tmp_path))
    metadata = FileSystemReader(str(tmp_path)).read_metadata()
    assert checkpoint._ep_partitioned_fqns(model) == {"expert.weight"}
    assert "optimizer.param_groups.0.params" in metadata.state_dict_metadata
    assert not any(key.startswith("optimizer.state.expert.weight::ep")
                   for key in metadata.state_dict_metadata)
    checkpoint._validate_optimizer_state(metadata, model, optimizer)
    assert not optimizer.state
    torch.testing.assert_close(model.expert.weight.grad, torch.ones_like(model.expert.weight),
                               rtol=0, atol=0)
