"""Native packed-policy admission on a real prepared TP/FSDP2 model."""

import copy
from datetime import timedelta
from functools import wraps

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor

from tests.engine.backends.fsdp2.test_tp_integration import PackedDenseDecoder, _batch
from tests.engine.backends.fsdp2.test_tensor_parallel import _native_plan
from torchturbo.engine import EngineConfig, TorchTurboEngine


def _record_calls(function, name, calls):
    @wraps(function)
    def recorded(*args, **kwargs):
        calls.append(name)
        return function(*args, **kwargs)

    return recorded


def _run_packed_rl_admission(rank, init_file, forward_only):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(57)
        config = EngineConfig.from_dict({"mixed_precision": "fp32"})
        config.activation_checkpointing.enabled = False
        engine = TorchTurboEngine(config)
        model = PackedDenseDecoder()
        reference = copy.deepcopy(model)
        plan = _native_plan()
        engine.model = engine.backend.prepare_model(model, plan)
        engine.parallel_plan = plan
        assert config.parallel_plan.tp_size == 1
        assert engine.get_tp_info()["tp_size"] == 2
        assert isinstance(model.layers[0].up.weight, DTensor)

        calls = []
        engine._compute_total_loss_weight = _record_calls(
            engine._compute_total_loss_weight, "loss_weight", calls,
        )
        engine._get_loss_multiplier = _record_calls(
            engine._get_loss_multiplier, "loss_multiplier", calls,
        )
        engine._backward_loss = _record_calls(
            engine._backward_loss, "backward", calls,
        )
        handle = model.register_forward_pre_hook(
            lambda _module, _args: calls.append("forward"),
        )

        def process_fn(logits, ctx, loss_fn):
            calls.append("process")
            return loss_fn(logits)

        def loss_fn(logits):
            calls.append("loss")
            return logits.square().mean()

        def loss_weight_fn(mb):
            calls.append("loss_weight_callback")
            return torch.tensor(1.0)

        def auxiliary_loss_fn(outputs, ctx, multiplier):
            calls.append("auxiliary_loss")
            return outputs.logits.sum() * 0.0

        batch = _batch(0, 0, 0)
        mb = {
            "orig_mb": {
                "input_ids": batch["input_ids"].flatten(),
                "cu_seqlens": batch["cu_seq_lens_q"],
            },
            "padded_mb": batch,
        }
        with pytest.raises(ValueError, match="packed RL.*TP"):
            engine.forward_backward_batch(
                [mb], process_fn=process_fn, loss_fn=loss_fn,
                loss_weight_fn=loss_weight_fn, forward_only=forward_only,
                auxiliary_loss_fn=auxiliary_loss_fn,
            )
        assert calls == []
        assert all(parameter.grad is None for parameter in model.parameters())
        handle.remove()

        # The independent no-grad helper does not execute a policy objective.
        logprobs = engine.compute_logprobs([mb], temperature=0.7)
        with torch.no_grad():
            logits = reference(**batch).logits[0]
            expected = (logits[[1, 3, 4]].float() / 0.7).log_softmax(dim=-1)
            expected = expected[torch.arange(3), batch["input_ids"][0, [2, 4, 5]]]
        torch.testing.assert_close(logprobs, expected, rtol=3e-5, atol=3e-6)
        assert not logprobs.requires_grad
        assert all(parameter.grad is None for parameter in model.parameters())
        assert calls == []
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("forward_only", [False, True])
def test_prepared_tp_rejects_packed_rl_before_execution(tmp_path, forward_only):
    mp.spawn(
        _run_packed_rl_admission,
        args=(str(tmp_path / "rendezvous"), forward_only), nprocs=2, join=True,
    )


def test_packed_rl_requires_prepared_model_before_tp_admission():
    engine = object.__new__(TorchTurboEngine)
    engine.model = None
    with pytest.raises(RuntimeError, match="prepare_model"):
        engine.forward_backward_batch(
            [], process_fn=None, loss_fn=None, loss_weight_fn=None,
        )
