"""Native CPU/Gloo TP layout norms, clipping, and optimizer groups."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from torchturbo.engine.backends.fsdp2.parallel_context import (
    ParallelContext,
    ParameterOwnership,
)


class SharedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 5, dtype=torch.float64)
        self.head = nn.Linear(5, 3, bias=False, dtype=torch.float64)
        self.head.weight = self.embedding.weight


class GradientModel(nn.Module):
    def __init__(self, out_features, num_experts, ep_rank):
        super().__init__()
        torch.manual_seed(11)
        self.dense = nn.Sequential(
            nn.Linear(5, out_features, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(out_features, 4, dtype=torch.float64),
        )
        self.norm = nn.LayerNorm(5, dtype=torch.float64)
        self.shared = SharedHead()
        self.replicated = SharedHead()
        torch.manual_seed(22 + ep_rank)
        self.experts = nn.Module()
        self.experts.weight1 = nn.Parameter(
            torch.randn(num_experts * 3, 2, 6, dtype=torch.float64)
        )
        self.experts.weight2 = nn.Parameter(
            torch.randn(num_experts, 6, 3, dtype=torch.float64)
        )


def _prepare_model(context, out_features, num_experts):
    ep_rank = context.ep_mesh.get_local_rank()
    model = GradientModel(out_features, num_experts, ep_rank)
    parallelize_module(model.dense, context.tp_mesh, {
        "0": ColwiseParallel(),
        "2": RowwiseParallel(input_layouts=Shard(-1)),
    })
    for module in (model.dense, model.norm, model.shared):
        fully_shard(module, mesh=context.dense_fsdp_mesh)
    replicated_weight = nn.Parameter(distribute_tensor(
        model.replicated.embedding.weight.detach(), context.tp_mesh,
        [Replicate()], src_data_rank=None,
    ))
    model.replicated.embedding.weight = replicated_weight
    model.replicated.head.weight = replicated_weight
    fully_shard(model.replicated, mesh=context.dense_fsdp_mesh)
    for name, placement in (("weight1", Shard(2)), ("weight2", Shard(1))):
        parameter = model.experts.get_parameter(name)
        model.experts.register_parameter(name, nn.Parameter(distribute_tensor(
            parameter.detach(), context.etp_mesh, [placement], src_data_rank=None,
        )))
    fully_shard(model.experts, mesh=context.expert_fsdp_mesh)
    ownership = ParameterOwnership()
    ownership.register(model, set(model.experts.parameters()))
    return model, ownership


def _gradient(name, shape, ep_rank, step=0):
    offset = sum(name.encode()) / 100 + step / 7
    if name.startswith("experts."):
        offset += ep_rank * 3
    return (torch.arange(shape.numel(), dtype=torch.float64).reshape(shape) + 1) / 17 + offset


def _inject(model, ep_rank, step=0):
    for name, parameter in model.named_parameters():
        parameter.grad = distribute_tensor(
            _gradient(name, parameter.shape, ep_rank, step),
            parameter.device_mesh, parameter.placements, src_data_rank=None,
        )


def _reference_norm(model, norm_type, step=0):
    gradients = []
    for name, parameter in model.named_parameters():
        if name.startswith("experts."):
            gradients.extend(_gradient(name, parameter.shape, ep, step) for ep in range(2))
        else:
            gradients.append(_gradient(name, parameter.shape, 0, step))
    # The reference concatenates complete logical gradients, independently of
    # the helper's scalar buckets and domain combination.
    return torch.linalg.vector_norm(torch.cat([grad.flatten() for grad in gradients]), norm_type)


def _group_options():
    return [
        {"lr": .003, "betas": (.8, .9), "eps": 1e-6, "weight_decay": .04,
         "amsgrad": True, "maximize": False, "foreach": True,
         "capturable": False, "differentiable": False, "fused": False,
         "initial_lr": .007},
        {"lr": .009, "betas": (.7, .95), "eps": 1e-5, "weight_decay": .02,
         "amsgrad": False, "maximize": True, "foreach": True,
         "capturable": False, "differentiable": False, "fused": False,
         "initial_lr": .012},
    ]


def _worker(rank, rendezvous, world_size, out_features, num_experts):
    from torchturbo.engine.backends.fsdp2.tp_gradient import (
        clip_grad_norm_by_layout_,
        split_optimizer_groups_by_layout,
    )

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        context = ParallelContext("cpu", sp_size=1, ep_size=2, tp_size=2, etp_size=2)
        model, ownership = _prepare_model(context, out_features, num_experts)
        ep_rank = context.ep_mesh.get_local_rank()
        kwargs = {"ownership": ownership, "ep_group": context.ep_mesh.get_group(),
                  "device": torch.device("cpu")}
        parameters = list(model.parameters())
        assert any(isinstance(placement, _StridedShard)
                   for parameter in parameters for placement in parameter.placements)
        if world_size == 8:
            any_empty = torch.tensor(any(parameter.to_local().numel() == 0 for parameter in parameters))
            dist.all_reduce(any_empty, op=dist.ReduceOp.MAX)
            assert any_empty.item()

        # Isolate a same-axis composed Colwise weight so another parameter's
        # larger maximum cannot hide a missing strided DP-axis reduction.
        _inject(model, ep_rank)
        colwise_weight = model.dense[0].weight
        expected_colwise = _gradient("dense.0.weight", colwise_weight.shape, 0).abs().max()
        actual_colwise = clip_grad_norm_by_layout_([colwise_weight], float("inf"), float("inf"), **kwargs)
        torch.testing.assert_close(actual_colwise, expected_colwise, rtol=1e-12, atol=1e-12)

        for norm_type in (1.0, 2.0, 3.0, float("inf")):
            _inject(model, ep_rank)
            expected = _reference_norm(model, norm_type)
            actual = clip_grad_norm_by_layout_(iter(parameters), float("inf"), norm_type, **kwargs)
            assert not isinstance(actual, DTensor)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.grad.full_tensor(), _gradient(name, parameter.shape, ep_rank))
            actual = clip_grad_norm_by_layout_(parameters, .25, norm_type, **kwargs)
            coefficient = (.25 / (expected + 1e-6)).clamp(max=1)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.grad.full_tensor(),
                    _gradient(name, parameter.shape, ep_rank) * coefficient, rtol=1e-12, atol=1e-12)

        for norm_type in (2.0, float("inf")):
            for nonfinite in (float("nan"), float("inf")):
                _inject(model, ep_rank)
                if rank == 0:
                    model.dense[0].weight.grad.to_local().flatten()[0] = nonfinite
                with pytest.raises(RuntimeError, match="Non-finite gradient norm"):
                    clip_grad_norm_by_layout_(parameters, .25, norm_type, **kwargs)

        _inject(model, ep_rank)
        model.norm.bias.grad = None
        expected_grads = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                owners = range(2) if ownership.is_expert(parameter) else (0,)
                expected_grads.extend(_gradient(name, parameter.shape, owner) for owner in owners)
        expected = torch.linalg.vector_norm(torch.cat([gradient.flatten() for gradient in expected_grads]))
        torch.testing.assert_close(clip_grad_norm_by_layout_(parameters, float("inf"), 2, **kwargs), expected)
        model.zero_grad(set_to_none=True)
        assert clip_grad_norm_by_layout_(parameters, 1, float("inf"), **kwargs).item() == 0
        assert clip_grad_norm_by_layout_(parameters, 1, 2, **kwargs).item() == 0

        options = _group_options()
        original_groups = [{"params": parameters[::2], **options[0]},
                           {"params": parameters[1::2], **options[1]}]
        model.norm.bias.requires_grad_(False)
        split_groups = split_optimizer_groups_by_layout(original_groups, ownership)
        cursor = 0
        for group, group_options in zip(original_groups, options):
            expected_buckets = {}
            for parameter in group["params"]:
                if parameter.requires_grad:
                    key = (ownership.is_expert(parameter), parameter.device_mesh, parameter.placements)
                    expected_buckets.setdefault(key, []).append(parameter)
            for expected_parameters in expected_buckets.values():
                split = split_groups[cursor]
                assert {key: value for key, value in split.items() if key != "params"} == group_options
                assert [id(parameter) for parameter in split["params"]] == [id(parameter) for parameter in expected_parameters]
                cursor += 1
        assert cursor == len(split_groups)
        assert original_groups[0]["params"] == parameters[::2]
        model.norm.bias.requires_grad_(True)
        split_groups = split_optimizer_groups_by_layout(original_groups, ownership)

        references = {name: nn.Parameter(parameter.full_tensor().detach().clone())
                      for name, parameter in model.named_parameters()}
        names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
        reference_groups = [{"params": [references[names_by_id[id(parameter)]] for parameter in group["params"]],
                             **group_options} for group, group_options in zip(original_groups, options)]
        optimizer = torch.optim.AdamW(split_groups)
        reference_optimizer = torch.optim.AdamW(reference_groups)
        for step in range(2):
            _inject(model, ep_rank, step)
            expected = _reference_norm(model, 2, step)
            clip_grad_norm_by_layout_(parameters, .25, 2, **kwargs)
            for name, reference in references.items():
                reference.grad = _gradient(name, reference.shape, ep_rank, step) * (.25 / (expected + 1e-6)).clamp(max=1)
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.full_tensor(), references[name], rtol=1e-12, atol=1e-12)
                for state_key in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(optimizer.state[parameter][state_key].full_tensor(),
                        reference_optimizer.state[references[name]][state_key], rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(optimizer.state[parameter]["step"], reference_optimizer.state[references[name]]["step"])
                if optimizer.state[parameter].keys() >= {"max_exp_avg_sq"}:
                    torch.testing.assert_close(optimizer.state[parameter]["max_exp_avg_sq"].full_tensor(),
                        reference_optimizer.state[references[name]]["max_exp_avg_sq"], rtol=1e-12, atol=1e-12)

        # Mesh equality cannot identify expert ownership. Exercise equal-layout
        # semantic buckets and refreshed aliases after assign=True replacement.
        equal_layout = nn.Module()
        equal_layout.dense = nn.Parameter(distribute_tensor(torch.ones(3), context.dense_fsdp_mesh, [Shard(0)], src_data_rank=None))
        equal_layout.expert = nn.Parameter(distribute_tensor(torch.ones(3), context.dense_fsdp_mesh, [Shard(0)], src_data_rank=None))
        ownership.register(equal_layout, {equal_layout.expert})
        assert len(split_optimizer_groups_by_layout([{"params": list(equal_layout.parameters())}], ownership)) == 2
        old_expert = equal_layout.expert
        equal_layout.load_state_dict(equal_layout.state_dict(), assign=True)
        ownership.bind(equal_layout)
        assert equal_layout.expert is not old_expert
        assert not ownership.is_expert(old_expert)
        assert ownership.is_expert(equal_layout.expert)
        assert len(split_optimizer_groups_by_layout([{"params": list(equal_layout.parameters())}], ownership)) == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size,out_features,num_experts", [(4, 6, 3), (8, 4, 1), (8, 12, 3)])
def test_native_tp_norm_clipping_and_adamw(tmp_path, world_size, out_features, num_experts):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}", world_size, out_features, num_experts),
             nprocs=world_size, join=True)


@pytest.mark.parametrize("norm_type", [0, -1, -float("inf"), float("nan")])
def test_reject_unsupported_norm_type(norm_type):
    from torchturbo.engine.backends.fsdp2.tp_gradient import clip_grad_norm_by_layout_

    with pytest.raises(ValueError, match="positive"):
        clip_grad_norm_by_layout_([], 1, norm_type, ownership=ParameterOwnership(),
                                 ep_group=None, device=torch.device("cpu"))
