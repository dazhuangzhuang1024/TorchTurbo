from torch import nn
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)

from ...parallel import TensorParallelPlanSpec, TensorParallelValidator, FSDPPlanSpec, FSDPParallelPlan
from torchturbo.engine.tensor_parallel.expert import GroupedMLPParallel
from torchturbo.engine.tensor_parallel.parallelize import (
    SharedHeadNormParallel,
)
from ..modules.moe.moe_layer import MoELayer, MoEConfig, MoESubmodules
from ..modules.moe.module import ModuleSpec
from ..modules.moe.experts import GroupedMLP
from ..modules.moe.config_converter import get_base_moe_config


class Qwen3MoeTensorParallelValidator(TensorParallelValidator):
    """Validate actual routed layers and complete projected attention heads."""

    def validate(self, model: nn.Module, parallel_plan: FSDPParallelPlan) -> None:
        tp_size, sp_size = parallel_plan.tp_size, parallel_plan.sp_size
        for index, layer in enumerate(model.model.layers):
            if not isinstance(layer.mlp, (Qwen3MoeSparseMoeBlock, MoELayer)):
                raise ValueError(
                    f"Qwen3-MoE TP requires all-routed layers; layer {index} has a dense MLP."
                )
            attention = layer.self_attn
            head_partition_width = attention.head_dim * tp_size
            for name, projection in (
                ("q_proj", attention.q_proj), ("k_proj", attention.k_proj),
                ("v_proj", attention.v_proj),
            ):
                if projection.out_features % head_partition_width:
                    raise ValueError(
                        f"{name}.out_features={projection.out_features} must be divisible by "
                        f"head_dim={attention.head_dim} * tp_size={tp_size}."
                    )
            if attention.attention_dropout != 0.0:
                raise ValueError("Qwen3-MoE TP requires attention_dropout=0.")
            for name, projection in (("k_proj", attention.k_proj), ("v_proj", attention.v_proj)):
                local_kv_heads = projection.out_features // head_partition_width
                repeated_kv_heads = local_kv_heads * max(1, sp_size // local_kv_heads)
                if repeated_kv_heads % sp_size:
                    raise ValueError(
                        f"Qwen3-MoE TP-local KV heads in {name} are incompatible with SP."
                    )


class QWen3MoeLayerWrapper(MoELayer):

    def __init__(
        self, original_module, hf_config, sp_info, ep_info, moe_submodules, **kwargs
    ):
        ep_group = ep_info["ep_group"]
        ep_size = ep_info["ep_size"]
        qwen3_config_kwargs = get_base_moe_config(
            hf_config=hf_config,
            ep_size=ep_size,
            num_moe_experts=hf_config.num_experts,
            params_dtype=hf_config.torch_dtype,
        )
        moe_config = MoEConfig(**qwen3_config_kwargs)

        super().__init__(
            config=moe_config,
            ep_group=ep_group,
            submodules=moe_submodules,
        )

        if hasattr(original_module, "shared_experts"):
            self.shared_experts = original_module.shared_experts

    def forward(self, hidden_states, input_ids=None):
        output, _ = super().forward(hidden_states, input_ids)
        return output


def get_parallel_plan() -> FSDPParallelPlan:
    from ...parallel import FSDPPlacementSpec

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            "model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "model.layers.*.mlp": FSDPPlanSpec(
                replace_module=QWen3MoeLayerWrapper,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "model.embed_tokens": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
            "lm_head": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
        },
        sp_size=1,
        ep_size=1,
        tp_size=1,
        etp_size=1,
        tensor_parallel_plan=TensorParallelPlanSpec(
            validator=Qwen3MoeTensorParallelValidator(),
            plan={
                "model.layers.*.self_attn.q_proj": ColwiseParallel(),
                "model.layers.*.self_attn.k_proj": ColwiseParallel(),
                "model.layers.*.self_attn.v_proj": ColwiseParallel(),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(input_layouts=Shard(-1)),
                "model.layers.*.self_attn.q_norm": SharedHeadNormParallel(head_axis=2),
                "model.layers.*.self_attn.k_norm": SharedHeadNormParallel(head_axis=2),
                "model.layers.*.mlp.experts": GroupedMLPParallel(),
            },
        ),
    )
