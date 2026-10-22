"""Aligned expert tensor parallelism at the shared local-kernel boundary."""

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle

from torchturbo.models.modules.moe.experts import GroupedMLP


class GroupedMLPParallel(ParallelStyle):
    """Shard every expert's paired gate/up features and sum complete outputs."""

    def validate(self, module: nn.Module, mesh_size: int) -> None:
        """Validate both full packed weights before any parameter is rebound."""
        if not isinstance(module, GroupedMLP):
            raise ValueError("GroupedMLPParallel requires the shared GroupedMLP.")
        if not module.config.gated_linear_unit:
            raise ValueError("ETP requires gated_linear_unit=True.")
        if module.use_te:
            raise ValueError("ETP requires use_te=False.")
        if module.config.moe_apply_probs_on_input:
            raise ValueError("ETP requires moe_apply_probs_on_input=False.")
        features = module.config.moe_ffn_hidden_size
        num_experts = module.num_local_experts
        hidden_size = module.config.hidden_size
        for name, value in (
            ("moe_ffn_hidden_size", features), ("hidden_size", hidden_size),
            ("num_local_experts", num_experts),
        ):
            if type(value) is not int:
                raise ValueError(f"ETP {name} ({value!r}) must be a plain integer.")
        if features % mesh_size:
            raise ValueError("ETP requires moe_ffn_hidden_size divisible by ETP size.")
        for name, expected in (
            ("weight1", (num_experts * hidden_size, 2 * features)),
            ("weight2", (num_experts * features, hidden_size)),
        ):
            actual = tuple(module.get_parameter(name).shape)
            if actual != expected:
                raise ValueError(f"ETP {name} shape {actual} must be {expected}.")

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self.validate(module, device_mesh.size())
        if device_mesh.size() == 1:
            return module
        num_experts = module.num_local_experts
        hidden_size = module.config.hidden_size
        features = module.config.moe_ffn_hidden_size
        for name, shape, placement in (
            ("weight1", (num_experts * hidden_size, 2, features), Shard(2)),
            ("weight2", (num_experts, features, hidden_size), Shard(1)),
        ):
            parameter = module.get_parameter(name)
            logical_weight = nn.Parameter(
                parameter.reshape(shape), requires_grad=parameter.requires_grad
            )
            # Initialize from the full source weight with native source-lane scatter.
            # Checkpoint loading instead slices already-broadcast full tensors.
            sharded_weight = distribute_tensor(logical_weight, device_mesh, [placement])
            module.register_parameter(
                name, nn.Parameter(sharded_weight, requires_grad=parameter.requires_grad)
            )

        def prepare_input(module: nn.Module, inputs: tuple) -> tuple:
            hidden_states = DTensor.from_local(
                inputs[0], device_mesh, [Replicate()], run_check=False
            ).to_local(grad_placements=[Partial()])
            return (hidden_states, *inputs[1:])

        def complete_output(module: nn.Module, inputs: tuple, output: tuple) -> tuple:
            hidden_states = DTensor.from_local(
                output[0], device_mesh, [Partial()], run_check=False
            ).redistribute(placements=[Replicate()]).to_local()
            return hidden_states, output[1]

        module.register_forward_pre_hook(prepare_input)
        module.register_forward_hook(complete_output)
        return module
