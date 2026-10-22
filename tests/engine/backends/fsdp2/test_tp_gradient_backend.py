"""TP backend dispatch and unchanged legacy norm and optimizer behavior."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.tensor import Shard, distribute_tensor

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.config import FSDPConfig
from torchturbo.engine.config import EngineConfig, MuonOptimizerConfig
from torchturbo.engine.optimizers import MuonPlan, MuonRecipe, OptimizerClassPlan
from torchturbo.engine.backends.fsdp2.parallel_context import (
    ParallelContext,
    ParameterOwnership,
)
from torchturbo.runtime.cpu import CpuRuntime

from .test_tp_gradient import _gradient, _inject, _prepare_model, _reference_norm


class CustomSGD(torch.optim.SGD):
    pass


def _backend(context, ownership=None):
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend._parallel_context = context
    backend._parameter_ownership = ownership or ParameterOwnership()
    backend._device_runtime = CpuRuntime()
    backend.fsdp_config = FSDPConfig()
    backend.config = EngineConfig()
    backend.gradient_scaling_factor = .5
    return backend


@pytest.mark.parametrize("tp_size,etp_size", [(2, 1), (1, 2)])
@pytest.mark.parametrize("optimizer_cls", [CustomSGD, torch.optim.Adam])
def test_tp_rejects_actual_non_adamw_plan(tp_size, etp_size, optimizer_cls):
    model = nn.Linear(3, 2)
    backend = _backend(SimpleNamespace(tp_size=tp_size, etp_size=etp_size))
    assert backend.config.optimizer.name != "muon"
    plan = OptimizerClassPlan(model, optimizer_cls, {"lr": .01, "weight_decay": .02})
    with pytest.raises(ValueError, match="only.*AdamW.*OptimizerClassPlan"):
        backend.prepare_optimizer(plan)
    assert model.weight.grad is None


@pytest.mark.parametrize("tp_size,etp_size", [(2, 1), (1, 2)])
def test_tp_rejects_actual_muon_plan_with_adamw_config(tp_size, etp_size):
    model = nn.Linear(3, 2)
    backend = _backend(SimpleNamespace(tp_size=tp_size, etp_size=etp_size))
    plan = MuonPlan(model, recipe=MuonRecipe({}),
                    config=MuonOptimizerConfig(lr=.01, weight_decay=.02))
    assert backend.config.optimizer.name != MuonPlan.name
    with pytest.raises(ValueError, match="only.*AdamW.*OptimizerClassPlan"):
        backend.prepare_optimizer(plan)


def test_legacy_custom_optimizer_and_nonpositive_norm_remain_accepted():
    model = nn.Linear(3, 2)
    backend = _backend(None)
    optimizer = backend.prepare_optimizer(OptimizerClassPlan(
        model, CustomSGD, {"lr": .01, "weight_decay": .02}
    ))
    assert type(optimizer) is CustomSGD
    parameter = nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([3., 4.])
    actual = backend.clip_grad_norm_([parameter], float("inf"), norm_type=0)
    expected = torch.linalg.vector_norm(torch.nn.utils.get_total_norm([parameter.grad], 0).unsqueeze(0), 0)
    torch.testing.assert_close(actual, expected)


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=4, timeout=timedelta(seconds=60))
    try:
        context = ParallelContext("cpu", sp_size=1, ep_size=2, tp_size=2, etp_size=2)
        model, ownership = _prepare_model(context, 6, 3)
        backend = _backend(context, ownership)
        ep_rank = context.ep_mesh.get_local_rank()
        parameters = list(model.parameters())
        for norm_type in (1., 2., 3., float("inf")):
            _inject(model, ep_rank)
            expected = _reference_norm(model, norm_type)
            actual = backend.clip_grad_norm_(iter(parameters), .25, norm_type)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            coefficient = (.25 / (expected + 1e-6)).clamp(max=1)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.grad.full_tensor(),
                    _gradient(name, parameter.shape, ep_rank) * coefficient,
                    rtol=1e-12, atol=1e-12)
        with pytest.raises(ValueError, match="positive"):
            backend.clip_grad_norm_(parameters, 1, 0)

        plan = OptimizerClassPlan(model, torch.optim.AdamW,
                                  {"lr": .003, "weight_decay": .04, "foreach": True})
        optimizer = backend.prepare_optimizer(plan)
        assert type(optimizer) is torch.optim.AdamW
        for group in optimizer.param_groups:
            layouts = {(ownership.is_expert(parameter), parameter.device_mesh, parameter.placements)
                       for parameter in group["params"]}
            assert len(layouts) == 1
            assert group["lr"] == .003 and group["foreach"] is True
        references = {name: nn.Parameter(parameter.full_tensor().detach().clone())
                      for name, parameter in model.named_parameters()}
        # Build reference groups with the same original decay/no-decay policy.
        reference_groups = []
        for group in plan.get_param_groups():
            reference_groups.append({"params": [references[name] for name in group.params_by_fqn],
                                     **group.optimizer_kwargs})
        reference_optimizer = torch.optim.AdamW(reference_groups, **plan.optimizer_kwargs)
        for step in range(2):
            _inject(model, ep_rank, step)
            expected = _reference_norm(model, 2, step)
            backend.clip_grad_norm_(parameters, .25, 2)
            for name, parameter in references.items():
                parameter.grad = _gradient(name, parameter.shape, ep_rank, step) * (.25 / (expected + 1e-6)).clamp(max=1)
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.full_tensor(), references[name], rtol=1e-12, atol=1e-12)
                for state_name in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(optimizer.state[parameter][state_name].full_tensor(),
                        reference_optimizer.state[references[name]][state_name], rtol=1e-12, atol=1e-12)

        legacy_context = ParallelContext("cpu", sp_size=1, ep_size=2)
        legacy = nn.Module()
        legacy.dense = nn.Parameter(distribute_tensor(torch.ones(4), legacy_context.root_mesh, [Shard(0)], src_data_rank=None))
        legacy.expert = nn.Parameter(distribute_tensor(torch.ones(4), legacy_context.expert_fsdp_mesh, [Shard(0)], src_data_rank=None))
        legacy.ignored = nn.Parameter(torch.ones(2))
        legacy.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
        legacy_ownership = ParameterOwnership()
        legacy_ownership.register(legacy, {legacy.expert})
        legacy_backend = _backend(legacy_context, legacy_ownership)
        supplied = [legacy.ignored, legacy.expert, legacy.frozen, legacy.dense]
        groups = legacy_backend._split_optimizer_groups_by_topology([{"params": supplied, "lr": .07}])
        assert len(groups) == 3
        assert groups[0]["params"][0] is legacy.dense
        assert groups[1]["params"][0] is legacy.expert
        assert groups[2]["params"][0] is legacy.ignored
        assert all(group["lr"] == .07 for group in groups)
        dense_gradient = torch.tensor([1., 2., 3., 4.])
        owner_rank = legacy_context.ep_mesh.get_local_rank()
        expert_gradient = torch.tensor([2., 4., 6., 8.]) + owner_rank
        legacy.dense.grad = distribute_tensor(dense_gradient, legacy.dense.device_mesh,
                                              legacy.dense.placements, src_data_rank=None)
        legacy.expert.grad = distribute_tensor(expert_gradient, legacy.expert.device_mesh,
                                               legacy.expert.placements, src_data_rank=None)
        legacy.ignored.grad = torch.tensor([3., 4.])
        expected = torch.linalg.vector_norm(torch.cat([
            dense_gradient, torch.tensor([3., 4.]),
            torch.tensor([2., 4., 6., 8.]) * .5,
            torch.tensor([3., 5., 7., 9.]) * .5,
        ]))
        actual = legacy_backend.clip_grad_norm_(supplied, .5, 2)
        torch.testing.assert_close(actual, expected)
        coefficient = (.5 / (expected + 1e-6)).clamp(max=1)
        torch.testing.assert_close(legacy.dense.grad.full_tensor(), dense_gradient * coefficient)
        torch.testing.assert_close(legacy.expert.grad.full_tensor(), expert_gradient * .5 * coefficient)
        torch.testing.assert_close(legacy.ignored.grad, torch.tensor([3., 4.]) * coefficient)
    finally:
        dist.destroy_process_group()


def test_native_backend_tp_and_legacy_routes(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True)
