"""Exact optimizer-step target normalization on CPU/Gloo."""

import os
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import EngineConfig, LossChunkConfig
from torchturbo.engine.core import TorchTurboEngine
from torchturbo.models.base import TurboModel
from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext


class PackedModel(TurboModel, nn.Module):
    def __init__(self, context=None):
        super().__init__()
        self.lm_head = nn.Linear(3, 7, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(21).reshape(7, 3) / 31)
        self.context = context
        self.prepare_calls = 0

    def on_prepare_inputs(self, inputs):
        self.prepare_calls += 1
        result = dict(inputs)
        result['labels'] = inputs['labels'].clone()
        if result['labels'][0, 4] != -100:
            result['labels'][0, 4] -= 1
        return result

    def get_lm_head(self):
        return self.lm_head

    def forward(self, input_ids, **kwargs):
        hidden = F.one_hot(input_ids % 3, 3).float()
        if self.context is not None:
            hidden = hidden.chunk(self.context.sp_size, dim=1)[self.context.get_sp_info()['sp_rank']]
        return SimpleNamespace(logits=self.lm_head(hidden))


def batch(data_rank, micro):
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0]])
    labels = ids.clone()
    if micro == 1 or (data_rank == 1 and micro == 0):
        labels.fill_(-100)
    elif data_rank == 1:
        labels[..., 4] = -100
    cu = torch.tensor([0, 1, 3, 6], dtype=torch.int32)
    return {'input_ids': ids, 'labels': labels, 'cu_seq_lens_q': cu,
            'cu_seq_lens_k': cu, 'max_length_q': 3, 'max_length_k': 3}


def make_engine(context=None, chunked=False):
    from torchturbo.ops import get_or_create_ops
    from torchturbo.runtime.cpu import CpuRuntime
    get_or_create_ops(device_runtime=CpuRuntime())
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend._parallel_context = context
    backend.config = EngineConfig(loss_chunk=LossChunkConfig(enabled=chunked))
    if context is None:
        backend.get_sp_info = lambda: {'sp_size': 1, 'sp_group': None, 'sp_rank': 0}
        backend.get_tp_info = lambda: {'tp_size': 1}
        backend.get_dp_info = lambda: {'dp_size': 1, 'dp_group': None}
    engine = TorchTurboEngine.__new__(TorchTurboEngine)
    engine.backend = backend
    engine.config = backend.config
    engine._record_micro = lambda inputs: None
    if chunked:
        def forward(model, inputs, use_chunked_loss):
            hidden = F.one_hot(inputs['input_ids'] % 3, 3).float()
            if context is not None:
                hidden = hidden.chunk(context.sp_size, 1)[context.get_sp_info()['sp_rank']]
            return SimpleNamespace(logits=None), [hidden]
        backend._forward_for_lm_loss = forward
    return engine


def test_full_targets_mask_boundaries_and_physical_tail():
    targets = FSDP2Backend.prepare_loss_targets(batch(0, 0)['labels'], batch(0, 0)['cu_seq_lens_q'])
    assert targets.tolist() == [[-100, 3, -100, 5, 6, -100, -100, -100]]


def test_step_preparation_calls_hook_once_without_forward_or_input_mutation():
    engine = make_engine()
    engine.get_tp_info = lambda: {"tp_size": 2}
    model = PackedModel()
    original = batch(0, 0)
    prepared, count = engine.prepare_loss_step(model, [original, batch(0, 1)], torch.device('cpu'))
    assert count == 3
    assert model.prepare_calls == 2
    assert all(value['input_ids'].device.type == 'cpu' for value in prepared)
    engine.compute_prepared_loss(model, prepared[0])
    assert model.prepare_calls == 2
    assert 'labels' in prepared[0] and original['labels'][0, 4] == 5
    assert prepared[0]['labels'][0, 4] == 4
    with pytest.raises(ValueError, match='no valid targets'):
        engine.prepare_loss_step(model, [batch(0, 1)], torch.device('cpu'))


def _worker(rank, rendezvous, chunked):
    torch.set_num_threads(1)
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo0' if sys.platform == 'darwin' else 'lo')
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=8, timeout=timedelta(seconds=60))
    try:
        context = ParallelContext('cpu', sp_size=2, tp_size=2, ep_size=1)
        engine = make_engine(context, chunked)
        model = PackedModel(context)
        data_rank = context.get_dp_info()['dp_rank']
        prepared, count = engine.prepare_loss_step(model, [batch(data_rank, i) for i in range(3)], torch.device('cpu'))
        assert count == 8
        assert model.prepare_calls == 3
        with pytest.raises(ValueError, match='no valid targets'):
            engine.prepare_loss_step(model, [batch(data_rank, 1)], torch.device('cpu'))
        zero_local_batch = batch(data_rank, 0)
        _, zero_local_count = engine.prepare_loss_step(model, [zero_local_batch], torch.device('cpu'))
        assert zero_local_count == 3
        # N=3 is deliberately not divisible by S=2. The direct API and the
        # prepared API share the same objective without legacy floor rounding.
        direct_loss, direct_outputs = engine.compute_loss(
            model, zero_local_batch, True, num_items_in_batch=zero_local_count
        )
        one_prepared = engine.prepare_loss_inputs(model, zero_local_batch)
        prepared_loss, prepared_outputs = engine.compute_prepared_loss(
            model, one_prepared, True, num_items_in_batch=zero_local_count
        )
        torch.testing.assert_close(direct_loss, prepared_loss)
        torch.testing.assert_close(direct_outputs.main_loss, prepared_outputs.main_loss)
        total_report = 0
        for inputs in prepared:
            loss, outputs = engine.compute_prepared_loss(model, inputs, True, num_items_in_batch=count, gradient_accumulation_steps=3)
            assert not outputs.main_loss.requires_grad
            assert 'labels' in inputs
            loss.backward()
            total_report += outputs.main_loss
        group = context.dense_fsdp_mesh.get_group()
        dist.all_reduce(model.lm_head.weight.grad, group=group)
        model.lm_head.weight.grad /= context.dense_fsdp_mesh.size()
        reference = PackedModel()
        sums = []
        # Assemble the oracle by literal packed sample ranges, independently of
        # the production target builder, excluding length-one samples and tail.
        for d in range(2):
            for i in range(3):
                inputs = batch(d, i)
                if inputs['labels'][0, 4] != -100:
                    inputs['labels'][0, 4] -= 1
                logits = reference(input_ids=inputs['input_ids']).logits
                for start, end in ((0, 1), (1, 3), (3, 6)):
                    targets = inputs['labels'][:, start + 1:end].clone()
                    sums.append(F.cross_entropy(logits[:, start:end - 1].reshape(-1, 7), targets.reshape(-1), ignore_index=-100, reduction='sum'))
        objective = sum(sums) / 8
        objective.backward()
        torch.testing.assert_close(total_report, objective.detach())
        torch.testing.assert_close(model.lm_head.weight.grad, reference.lm_head.weight.grad)
        torch.optim.SGD(model.parameters(), lr=.1).step()
        torch.optim.SGD(reference.parameters(), lr=.1).step()
        torch.testing.assert_close(model.lm_head.weight, reference.lm_head.weight)
        with pytest.raises(ValueError, match='step-level'):
            engine.compute_loss(model, batch(d, 0))
        for invalid in (0, -1, torch.tensor(0)):
            with pytest.raises(ValueError, match='positive'):
                engine.compute_loss(model, batch(d, 0), num_items_in_batch=invalid)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('chunked', [False, True])
def test_distributed_step_loss_and_update(tmp_path, chunked):
    mp.spawn(_worker, args=(f'file://{tmp_path / "rendezvous"}', chunked), nprocs=8, join=True)


class NativeSFTModel(TurboModel, nn.Module):
    """Wrap the native TP decoder fixture in the engine output contract."""

    def __init__(self):
        super().__init__()
        from tests.engine.backends.fsdp2.test_tensor_parallel import DenseDecoder
        decoder = DenseDecoder()
        self.config = decoder.config
        self.embed = decoder.embed
        self.layers = decoder.layers
        self.norm = decoder.norm
        self.head = decoder.head

    def get_input_embeddings(self):
        return self.embed

    def get_lm_head(self):
        return self.head

    def get_norm(self):
        return self.norm

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for block in self.layers:
            x = block(x)
        from transformers.modeling_outputs import CausalLMOutput
        return CausalLMOutput(logits=self.head(self.norm(x)))


def _native_worker(rank, rendezvous, chunked):
    import copy
    from tests.engine.backends.fsdp2.test_tensor_parallel import _native_plan
    from torchturbo.engine.config import MixedPrecisionPolicy
    from torchturbo.runtime.cpu import CpuRuntime
    from torchturbo.ops import get_or_create_ops
    from torch.distributed.tensor import DTensor

    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo0' if sys.platform == 'darwin' else 'lo')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(51)
        model = NativeSFTModel()
        reference = copy.deepcopy(model)
        config = EngineConfig(mixed_precision=MixedPrecisionPolicy.FP32, loss_chunk=LossChunkConfig(enabled=chunked))
        backend = FSDP2Backend(config, CpuRuntime())
        backend.prepare_model(model, _native_plan())
        engine = TorchTurboEngine.__new__(TorchTurboEngine)
        engine.backend = backend
        engine.config = config
        engine._record_micro = lambda inputs: None
        get_or_create_ops(device_runtime=CpuRuntime())
        data_rank = rank // 2
        inputs = [batch(data_rank, i) for i in range(3)]
        prepared, count = engine.prepare_loss_step(model, inputs, torch.device('cpu'))
        assert count == 8
        report = 0
        for micro in prepared:
            loss, outputs = engine.compute_prepared_loss(model, micro, True, num_items_in_batch=count, gradient_accumulation_steps=3)
            loss.backward()
            report += outputs.main_loss
        reference_sum = 0
        for d in range(2):
            for i in range(3):
                micro = batch(d, i)
                logits = reference(**micro).logits
                for start, end in ((0, 1), (1, 3), (3, 6)):
                    reference_sum += F.cross_entropy(logits[:, start:end-1].reshape(-1, 24), micro['labels'][:, start+1:end].reshape(-1), ignore_index=-100, reduction='sum')
        objective = reference_sum / count
        objective.backward()
        torch.testing.assert_close(report, objective.detach(), rtol=2e-5, atol=2e-5)
        for name, parameter in model.named_parameters():
            gradient = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
            torch.testing.assert_close(gradient, reference.get_parameter(name).grad, rtol=3e-5, atol=3e-5)
        torch.optim.AdamW(model.parameters(), lr=.001).step()
        torch.optim.AdamW(reference.parameters(), lr=.001).step()
        for name, parameter in model.named_parameters():
            value = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            torch.testing.assert_close(value, reference.get_parameter(name), rtol=3e-5, atol=3e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('chunked', [False, True])
def test_native_fsdp_tp_ce_accumulation_and_adamw(tmp_path, chunked):
    mp.spawn(_native_worker, args=(f'file://{tmp_path / "native"}', chunked), nprocs=4, join=True)
