"""Native TP plans and numerical FSDP2 composition on CPU/Gloo."""

import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, PrepareModuleInput, RowwiseParallel

from torchturbo.engine.model_fqn import build_model_fqn_map
from torchturbo.parallel import (
    TensorParallelPlanSpec,
    FSDPParallelPlan,
    FSDPPlacementSpec,
    FSDPPlanSpec,
)


class HeadAttention(nn.Module):
    """Small attention module with explicit local head metadata."""

    def __init__(self):
        super().__init__()
        self.num_heads = 8
        self.num_key_value_heads = 4
        self.head_dim = 2
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 8, bias=False)
        self.v_proj = nn.Linear(16, 8, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=False)
        self.q_norm = nn.RMSNorm(2)
        self.k_norm = nn.RMSNorm(2)

    def forward(self, x):
        shape = x.shape[:2]
        q = self.q_norm(self.q_proj(x).view(*shape, self.num_heads, 2)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(*shape, self.num_key_value_heads, 2)).transpose(1, 2)
        v = self.v_proj(x).view(*shape, self.num_key_value_heads, 2).transpose(1, 2)
        repeats = self.num_heads // self.num_key_value_heads
        y = nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1),
            is_causal=True,
        )
        return self.o_proj(y.transpose(1, 2).reshape(*shape, self.num_heads * 2))


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(16)
        self.attn = HeadAttention()
        self.ffn_norm = nn.RMSNorm(16)
        self.gate = nn.Linear(16, 32, bias=False)
        self.up = nn.Linear(16, 32, bias=False)
        self.down = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        h = self.ffn_norm(x)
        return x + self.down(nn.functional.silu(self.gate(h)) * self.up(h))


class DenseDecoder(nn.Module):
    def __init__(self, *, tied=False):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=4)
        self.embed = nn.Embedding(24, 16)
        self.layers = nn.ModuleList([DecoderBlock()])
        self.norm = nn.RMSNorm(16)
        self.head = nn.Linear(16, 24, bias=False)
        if tied:
            self.head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def get_lm_head(self):
        return self.head

    def get_norm(self):
        return self.norm

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.layers:
            x = block(x)
        return self.head(self.norm(x))


class DecoderBackbone(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.embed = decoder.embed
        self.layers = decoder.layers
        self.norm = decoder.norm

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.layers:
            x = block(x)
        return self.norm(x)


class ParentDenseDecoder(DenseDecoder):
    """Expose the tied embedding inside a callable parent FSDP boundary."""

    def __init__(self):
        super().__init__(tied=True)
        self.base = DecoderBackbone(self)
        del self.embed, self.layers, self.norm

    def get_input_embeddings(self):
        return self.base.embed

    def get_norm(self):
        return self.base.norm

    def forward(self, input_ids):
        return self.head(self.base(input_ids))


def _native_plan():
    from torchturbo.engine.tensor_parallel.parallelize import (
        PrepareModuleMetadata,
        SharedHeadNormParallel,
    )

    return FSDPParallelPlan(tp_size=2, tensor_parallel_plan=TensorParallelPlanSpec(plan={
        "layers.*.attn": PrepareModuleMetadata(divide=("num_heads", "num_key_value_heads")),
        "layers.*.attn.q_proj": ColwiseParallel(),
        "layers.*.attn.k_proj": ColwiseParallel(),
        "layers.*.attn.v_proj": ColwiseParallel(),
        "layers.*.attn.o_proj": RowwiseParallel(input_layouts=Shard(-1)),
        "layers.*.attn.q_norm": SharedHeadNormParallel(head_axis=2),
        "layers.*.attn.k_norm": SharedHeadNormParallel(head_axis=2),
        "layers.*.gate": ColwiseParallel(),
        "layers.*.up": ColwiseParallel(),
        "layers.*.down": RowwiseParallel(input_layouts=Shard(-1)),
    }))


def _run_native_tensor_test(rank, init_file):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import EngineConfig, MixedPrecisionPolicy
    from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext
    from torchturbo.engine.tensor_parallel.parallelize import (
        apply_tensor_parallel_plan,
        resolve_tensor_parallel_plan,
    )
    from torchturbo.runtime.cpu import CpuRuntime

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=4, timeout=timedelta(seconds=90))
    try:
        context = ParallelContext("cpu", sp_size=1, ep_size=1, tp_size=2, etp_size=1)
        assert context.root_mesh.mesh_dim_names == ("dp",)
        assert context.dense_fsdp_mesh.mesh.tolist() == ([0, 2] if rank % 2 == 0 else [1, 3])
        assert context.tp_mesh.mesh.tolist() == ([0, 1] if rank < 2 else [2, 3])
        assert context.get_dp_info()["dp_rank"] == rank // 2
        assert context.get_dp_info()["dp_size"] == 2
        assert context.get_tp_info()["tp_rank"] == rank % 2
        assert context.get_etp_info()["etp_size"] == 1
        for mesh in (context.dense_fsdp_mesh, context.tp_mesh, context.data_dp_mesh, context.sp_mesh):
            assert mesh._get_root_mesh() is context.root_mesh

        cases = (
            (False, False, False, False), (True, False, False, False),
            (True, "both", False, False), (True, "head", False, False),
            (True, "embed", False, False), (False, False, True, False),
            (True, False, False, True), (True, "head", False, True),
        )
        for tied, explicit, assigned, parent in cases:
            torch.manual_seed(13)
            model = ParentDenseDecoder() if parent else DenseDecoder(tied=tied)
            backbone = model.base if parent else model
            with torch.no_grad():
                backbone.layers[0].attn.q_norm.weight.copy_(torch.tensor([0.7, 1.3]))
                backbone.layers[0].attn.k_norm.weight.copy_(torch.tensor([1.1, 0.9]))
            reference = copy.deepcopy(model)
            backend = FSDP2Backend(EngineConfig(mixed_precision=MixedPrecisionPolicy.FP32), CpuRuntime())
            plan = _native_plan()
            if parent:
                plan.tensor_parallel_plan.plan = {
                    f"base.{pattern}": style
                    for pattern, style in plan.tensor_parallel_plan.plan.items()
                }
                plan.plan["base"] = FSDPPlanSpec(standalone_world_shard=True)
            if explicit:
                for fqn in ("embed", "head") if explicit == "both" else (explicit,):
                    plan.plan[fqn] = FSDPPlanSpec(standalone_world_shard=True)
            prepared = backend.prepare_model(model, plan)
            assert prepared is model
            assert model.config.num_attention_heads == 8
            assert backbone.layers[0].attn.num_heads == 4
            assert backbone.layers[0].attn.num_key_value_heads == 2
            assert backend._world_mesh.mesh_dim_names == ("dp",)
            assert backend._world_mesh.size() == 2
            assert backend.get_parallel_context()["tp"]["tp_size"] == 2
            if tied:
                assert model.head.weight is model.get_input_embeddings().weight
            if assigned:
                old = backbone.layers[0].attn.q_norm.weight
                model.load_state_dict({name: value.clone() for name, value in model.state_dict().items()}, assign=True)
                backend.finalize_model_load(model)
                assert backbone.layers[0].attn.q_norm.weight is not old
            assert backbone.layers[0].attn.q_norm.weight.placements[-1] == Replicate()
            for parameter in (model.get_input_embeddings().weight, model.get_norm().weight, model.head.weight):
                assert parameter.device_mesh.mesh_dim_names == ("dp",)

            # Different DP samples, replicated within each TP lane pair.
            inputs = [torch.tensor([[0, 2, 3], [5, 1, 7]]), torch.tensor([[9, 4, 8], [6, 3, 2]])]
            local_input = inputs[rank // 2]
            target = torch.arange(144, dtype=torch.float32).view(2, 3, 24) / 100
            expected = reference(local_input)
            actual = model(local_input)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            oracle_loss = sum((reference(ids) - target).square().mean() for ids in inputs) / 2
            oracle_loss.backward()
            (actual - target).square().mean().backward()
            for name, parameter in model.named_parameters():
                assert isinstance(parameter, DTensor)
                assert parameter.grad is not None, name
                assert isinstance(parameter.grad, DTensor), name
                assert parameter.grad.device_mesh == parameter.device_mesh, name
                assert parameter.grad.placements == parameter.placements, name
                torch.testing.assert_close(parameter.grad.full_tensor(), reference.get_parameter(name).grad, atol=2e-5, rtol=2e-5, msg=name)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.003)
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.full_tensor(), reference.get_parameter(name), atol=2e-5, rtol=2e-5, msg=name)
            if tied:
                assert model.head.weight is model.get_input_embeddings().weight

            # A cached backend must not apply a new model using the old mesh.
            for degree in (4, 1):
                unchanged = DenseDecoder()
                with pytest.raises(ValueError, match="initialized.*degrees"):
                    backend.prepare_model(unchanged, FSDPParallelPlan(tp_size=degree, tensor_parallel_plan=plan.tensor_parallel_plan))
                assert unchanged.layers[0].attn.num_heads == 8
                assert not isinstance(unchanged.layers[0].up.weight, DTensor)

        # TP replacements receive the same context later used for styles.
        def replace_projection(original_module, *, parallel_context, **kwargs):
            replacement = copy.deepcopy(original_module)
            replacement.tp_context = parallel_context
            return replacement

        replaced = DenseDecoder()
        plan = _native_plan()
        plan.plan["layers.0.up"] = FSDPPlanSpec(replace_module=replace_projection)
        backend = FSDP2Backend(EngineConfig(mixed_precision=MixedPrecisionPolicy.FP32), CpuRuntime())
        backend.prepare_model(replaced, plan)
        assert replaced.layers[0].up.tp_context["tp"]["tp_size"] == 2
        assert replaced.layers[0].up.tp_context["tp"]["tp_group"] is backend.get_tp_info()["tp_group"]
        assert isinstance(replaced.layers[0].up.weight, DTensor)

        # Validate all metadata targets before the first native style mutates weights.
        bad = DenseDecoder()
        bad.layers[0].attn.num_key_value_heads = 3
        invalid = FSDPParallelPlan(tp_size=2, tensor_parallel_plan=TensorParallelPlanSpec(plan={
            "layers.*.up": ColwiseParallel(),
            **_native_plan().tensor_parallel_plan.plan,
        }))
        resolved = resolve_tensor_parallel_plan(invalid.tensor_parallel_plan.plan, build_model_fqn_map(bad))
        with pytest.raises(ValueError, match="num_key_value_heads"):
            apply_tensor_parallel_plan(bad, resolved, tp_mesh=context.tp_mesh)
        assert not isinstance(bad.layers[0].up.weight, DTensor)
        assert bad.layers[0].attn.num_heads == 8

        for target in ("embed", "head"):
            unchanged = DenseDecoder()
            plan = _native_plan()
            plan.tensor_parallel_plan.plan[target] = ColwiseParallel()
            backend = FSDP2Backend(EngineConfig(mixed_precision=MixedPrecisionPolicy.FP32), CpuRuntime())
            with pytest.raises(ValueError, match="vocabulary"):
                backend.prepare_model(unchanged, plan)
            assert unchanged.layers[0].attn.num_heads == 8
            assert not isinstance(unchanged.layers[0].up.weight, DTensor)

        # Native parent input preparation and child projection plans both run.
        parent = nn.Sequential(nn.Linear(16, 32))
        plan = FSDPParallelPlan(tp_size=2, tensor_parallel_plan=TensorParallelPlanSpec(plan={
            "": PrepareModuleInput(input_layouts=Replicate(), desired_input_layouts=Replicate()),
            "0": ColwiseParallel(),
        }))
        apply_tensor_parallel_plan(parent, resolve_tensor_parallel_plan(plan.tensor_parallel_plan.plan, build_model_fqn_map(parent)), tp_mesh=context.tp_mesh)
        assert parent(torch.ones(2, 16)).shape == (2, 16)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_native_tp_decoder_forward_gradients_and_adamw(tmp_path):
    mp.spawn(_run_native_tensor_test, args=(str(tmp_path / "init"),), nprocs=4, join=True)


@pytest.mark.parametrize("feature", ["peft", "muon", "offload", "mtp", "vision_balance", "integer_reshard", "integer_mesh_dim", "world_mesh", "expert_domain", "etp", "dropout", "diffusion", "dspark", "sequence_parallel", "no_shard"])
def test_backend_rejects_unsupported_tp_before_replacement(feature, monkeypatch):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import EngineConfig, OptimizerConfig
    from torchturbo.parallel import VisionBalanceMode, VisionBalanceSpec

    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.config = EngineConfig()
    backend.fsdp_config = backend.config.backend_config
    backend._parallel_context = None
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    model = DenseDecoder()
    plan = _native_plan()
    peft = None
    if feature == "peft":
        peft = object()
    elif feature == "muon":
        backend.config.optimizer = OptimizerConfig(name="muon")
    elif feature == "offload":
        backend.fsdp_config.cpu_offload.enabled = True
    elif feature == "mtp":
        backend.config.mtp.enabled = True
    elif feature == "vision_balance":
        plan.plan["layers.0"] = FSDPPlanSpec(vision_balance=VisionBalanceSpec(mode=VisionBalanceMode.PATCH))
    elif feature == "integer_reshard":
        backend.fsdp_config.reshard_after_forward = 1
    elif feature == "integer_mesh_dim":
        plan.plan["layers.0"] = FSDPPlanSpec(placement=FSDPPlacementSpec(Shard(0), 0))
    elif feature == "world_mesh":
        plan.plan["layers.0"] = FSDPPlanSpec(placement=FSDPPlacementSpec(Shard(0), "world"))
    elif feature == "expert_domain":
        plan.plan["layers.0"] = FSDPPlanSpec(placement=FSDPPlacementSpec(Shard(0), "edp"))
    elif feature == "etp":
        plan.etp_size = 2
    elif feature == "dropout":
        model.dropout = nn.Dropout(0.1)
    elif feature == "diffusion":
        from torchturbo.models.base import TurboModel

        class DiffusionDecoder(DenseDecoder, TurboModel):
            is_diffusion_model = True

        model = DiffusionDecoder()
    elif feature == "dspark":
        model.config.dspark_block_size = 8
    elif feature == "no_shard":
        plan.plan["norm"] = FSDPPlanSpec(no_shard=True)
    elif feature == "sequence_parallel":
        from torch.distributed.tensor.parallel import SequenceParallel

        plan.tensor_parallel_plan.plan["norm"] = SequenceParallel()
    message = {
        "peft": "PEFT", "muon": "Muon", "offload": "offload", "mtp": "MTP",
        "vision_balance": "vision balance", "integer_reshard": "boolean",
        "integer_mesh_dim": "mesh_dim", "world_mesh": "WORLD",
        "expert_domain": "expert", "etp": "ETP", "dropout": "dropout",
        "diffusion": "diffusion", "dspark": "DSpark", "sequence_parallel": "sequence-sharded",
        "no_shard": "no_shard",
    }[feature]
    with pytest.raises(ValueError, match=message):
        backend.prepare_model(model, plan, peft_config=peft)
    assert model.layers[0].attn.num_heads == 8
    assert not isinstance(model.layers[0].up.weight, DTensor)


@pytest.mark.parametrize("tp_size,etp_size", [(2, 1), (1, 2), (2, 2)])
def test_tp_preparation_requires_initialized_process_group_before_replacement(monkeypatch, tp_size, etp_size):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.config import EngineConfig
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.config = EngineConfig()
    backend.fsdp_config = backend.config.backend_config
    backend._parallel_context = None
    model = DenseDecoder()
    original_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    original_identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    replacements = []

    def replace_modules(*args):
        replacements.append(args)
        pytest.fail("TP reached replacement without an initialized process group")

    monkeypatch.setattr(backend, "_replace_modules", replace_modules)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    plan = _native_plan()
    plan.tp_size = tp_size
    plan.etp_size = etp_size
    with pytest.raises(ValueError, match="TP.*initialized process group"):
        backend.prepare_model(model, plan)
    assert replacements == []
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == original_identities
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original_parameters[name], rtol=0, atol=0)
