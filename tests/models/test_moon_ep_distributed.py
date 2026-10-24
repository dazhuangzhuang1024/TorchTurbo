"""Virtual expert output and gradient parity through real EP and FSDP collectives."""

import os
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import EngineConfig
from torchturbo.engine.peft.lora import GroupedMLPLoRA
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.parallel import FSDPPlanSpec
from torchturbo.runtime.cpu import CpuRuntime
import torchturbo.runtime.runtime as runtime_module


def _check_moon(
    ep_group, *, empty="none", boundary="none", reentrant=False,
    freeze_inputs=False, freeze_experts=False, on_input=False, zero_probs=False,
    fsdp_mesh=None, mixed_precision=False, cast_inputs=False,
):
    ep_size = dist.get_world_size(ep_group)
    rank = dist.get_rank(ep_group)
    torch.manual_seed(172 + rank)
    config = MoEConfig(
        hidden_size=8, moe_ffn_hidden_size=4, num_moe_experts=ep_size * 2,
        expert_model_parallel_size=ep_size, moe_router_topk=1 if on_input else 2,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        moe_router_dtype="fp32", moe_apply_probs_on_input=on_input,
        moe_router_score_function="sigmoid" if on_input else "softmax",
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    reference = MoELayer(config, ep_group, submodules=submodules)
    candidate = MoELayer(
        replace(config, moe_token_dispatcher_type="moon"), ep_group,
        submodules=submodules,
    )
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.15)
        # Positive inputs select only rank zero's experts; negative inputs move
        # the hotspot to another owner, changing placement between forwards.
        reference.router.weight.zero_()
        for expert in range(config.moe_router_topk):
            reference.router.weight[expert].fill_(0.2 / (expert + 1))
    candidate.load_state_dict(reference.state_dict(), strict=True)
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    for name, value in candidate.state_dict().items():
        assert value.shape == reference.state_dict()[name].shape
    if freeze_inputs:
        reference.router.requires_grad_(False)
        candidate.router.requires_grad_(False)
    if freeze_experts is True or (freeze_experts == "one" and rank == 0):
        reference.experts.requires_grad_(False)
        candidate.experts.requires_grad_(False)
    if zero_probs:
        for layer in (reference, candidate):
            original_router = layer.router.forward

            def masked_router(*args, router=original_router):
                routing, logits = router(*args)
                first_expert = routing.expert_indices[:, :1]
                selected = routing.probs.gather(1, first_expert)
                zero_scores = selected - selected.detach()
                probs = routing.probs.scatter(1, first_expert, zero_scores)
                return replace(routing, probs=probs), logits

            layer.router.forward = masked_router

    if boundary != "none":
        backend = SimpleNamespace(
            config=EngineConfig.from_dict({
                "activation_checkpointing": {"enabled": True, "use_reentrant": reentrant},
            }),
            prefetch_manager=SimpleNamespace(
                suspend_expert_forward_prefetch=lambda model: nullcontext(),
            ),
        )
        key = "" if boundary == "whole" else "experts"
        plan = SimpleNamespace(modules={key: FSDPPlanSpec(activation_checkpoint=True)})
        FSDP2Backend._apply_activation_checkpointing(backend, candidate, plan)
    if fsdp_mesh is not None:
        policy = MixedPrecisionPolicy()
        if mixed_precision:
            policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                cast_forward_inputs=cast_inputs,
            )
            reference.experts.to(torch.bfloat16)
            if cast_inputs:
                def cast_reference_inputs(module, args):
                    tokens, counts, probs = args
                    return tokens.to(torch.bfloat16), counts, probs.to(torch.bfloat16)

                reference.experts.register_forward_pre_hook(cast_reference_inputs)
        fully_shard(candidate.experts, mesh=fsdp_mesh["edp"], mp_policy=policy)
        fully_shard(candidate, mesh=fsdp_mesh)

    lengths = [3 + rank, 5 + rank]
    if empty == "all" or (empty == "one" and rank == 0):
        lengths = [0, 0]
    elif empty == "tiny":
        lengths = [1, 1] if rank == 0 else [0, 0]
    dtype = torch.bfloat16 if mixed_precision else torch.float32
    inputs = [
        ((torch.rand(1, length, 8, dtype=dtype) + 1) * sign).requires_grad_(not freeze_inputs)
        for length, sign in zip(lengths, (1, -1), strict=True)
    ]
    ref_inputs = [x.detach().clone().requires_grad_(not freeze_inputs) for x in inputs]
    actual = []
    for index, x in enumerate(inputs):
        actual.append(candidate(x)[0])
        base_length = (3, 5)[index]
        source_lengths = [base_length + source for source in range(ep_size)]
        if empty == "one":
            source_lengths[0] = 0
        elif empty == "all":
            source_lengths = [0] * ep_size
        elif empty == "tiny":
            source_lengths = [1] + [0] * (ep_size - 1)
        total_routes = sum(source_lengths) * config.moe_router_topk
        target = total_routes // ep_size + (rank < total_routes % ep_size)
        counts = candidate.token_dispatcher.tokens_per_expert
        assert counts.numel() == 4
        assert counts.sum().item() == target
        if index == 0 and rank > 0 and target > 0:
            assert counts[2:].sum().item() > 0
    expected = [reference(x)[0] for x in ref_inputs]
    context = (
        f"rank={rank}, EP={ep_size}, empty={empty}, boundary={boundary}, "
        f"reentrant={reentrant}, frozen_inputs={freeze_inputs}, frozen_experts={freeze_experts}, "
        f"on_input={on_input}, zero_probs={zero_probs}, FSDP={fsdp_mesh is not None}"
    )
    tolerance = {"atol": 3e-3, "rtol": 3e-2} if mixed_precision else {"atol": 2e-6, "rtol": 2e-5}
    for result, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, wanted, **tolerance, msg=context)

    # Backward in reverse order, retaining both different plans until used.
    for result in reversed(actual):
        result.square().sum().backward()
    for result in reversed(expected):
        result.square().sum().backward()
    for result, wanted in zip(inputs, ref_inputs, strict=True):
        torch.testing.assert_close(result.grad, wanted.grad, **tolerance, msg=context)

    ref_parameters = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        gradient = parameter.grad
        wanted = ref_parameters[name].grad
        if parameter.requires_grad:
            assert gradient is not None and wanted is not None, (context, name)
        if fsdp_mesh is not None and parameter.requires_grad:
            assert gradient is not None and wanted is not None, (context, name)
            gradient = gradient.full_tensor()
            reduction_group = (
                fsdp_mesh["edp"].get_group() if name.startswith("experts.")
                else dist.group.WORLD
            )
            dist.all_reduce(wanted, group=reduction_group)
            wanted /= dist.get_world_size(reduction_group)
        torch.testing.assert_close(
            gradient, wanted, **tolerance, msg=f"{context}, {name}",
            check_dtype=not mixed_precision,
        )

    torch.optim.SGD(candidate.parameters(), lr=0.05).step()
    torch.optim.SGD(reference.parameters(), lr=0.05).step()
    for name, parameter in candidate.named_parameters():
        actual_weight = parameter.full_tensor() if fsdp_mesh is not None else parameter
        torch.testing.assert_close(
            actual_weight, ref_parameters[name], **tolerance, msg=f"{context}, updated {name}",
            check_dtype=not mixed_precision,
        )


def _run_moon(rank, world_size, init_file, fsdp):
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        runtime = CpuRuntime()
        runtime_module._SELECTED_RUNTIME = runtime
        get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)
        if fsdp:
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("edp", "ep"))
            group = mesh["ep"].get_group()
            for boundary in ("whole", "experts"):
                for reentrant in (False, True):
                    for empty in ("none", "one", "all"):
                        _check_moon(
                            group, fsdp_mesh=mesh, freeze_inputs=True,
                            boundary=boundary, reentrant=reentrant, empty=empty,
                        )
                        dist.barrier()
            _check_moon(group, fsdp_mesh=mesh, freeze_inputs=True, mixed_precision=True)
            _check_moon(
                group, fsdp_mesh=mesh, freeze_inputs=True,
                mixed_precision=True, cast_inputs=True,
            )
            for boundary, reentrant in (("none", False), ("experts", False), ("experts", True)):
                _check_moon(group, fsdp_mesh=mesh, boundary=boundary, reentrant=reentrant)
                dist.barrier()
        else:
            group = dist.group.WORLD
            for boundary in ("none", "whole", "experts"):
                modes = (False,) if boundary == "none" else (False, True)
                for reentrant in modes:
                    for empty in ("none", "one", "all"):
                        _check_moon(group, boundary=boundary, reentrant=reentrant, empty=empty)
                        dist.barrier()
            for options in (
                {"on_input": True}, {"zero_probs": True}, {"freeze_experts": True},
                {"freeze_experts": "one"}, {"empty": "tiny", "on_input": True},
                {"freeze_inputs": True, "boundary": "experts", "reentrant": True},
            ):
                _check_moon(group, **options)
                dist.barrier()
    finally:
        dist.destroy_process_group()


def _spawn_moon(world_size, init_file, fsdp):
    context = mp.spawn(
        _run_moon, args=(world_size, str(init_file), fsdp), nprocs=world_size, join=False,
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Moon EP workers did not complete within 120 seconds")
    finally:
        # A mismatched collective can hang Gloo's teardown after its timeout.
        # Keep regressions bounded even when a worker cannot unwind normally.
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_moon_outputs_gradients_and_checkpoint(ep_size, tmp_path):
    _spawn_moon(ep_size, tmp_path / "moon-ep", False)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_moon_fsdp_checkpoint_and_mixed_precision(tmp_path):
    _spawn_moon(4, tmp_path / "moon-fsdp", True)


def test_moon_rejects_te_and_expert_lora():
    config = MoEConfig(
        hidden_size=8, moe_ffn_hidden_size=4, moe_token_dispatcher_type="moon",
    )
    with pytest.raises(ValueError, match="Moon.*TE"):
        GroupedMLP(2, True, config)
    experts = GroupedMLP(2, False, config)
    with pytest.raises(ValueError, match="Moon.*LoRA"):
        GroupedMLPLoRA(experts, "default", r=2)
