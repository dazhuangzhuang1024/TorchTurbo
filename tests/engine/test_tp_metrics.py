"""Useful-model TP MFU and literal logical sampler accounting."""

from types import SimpleNamespace

import pytest
import torch

from torchturbo.engine.config import EngineConfig, MfuConfig
from torchturbo.engine.core import TorchTurboEngine
from torchturbo.engine.metrics.mfu_estimator import MFUEstimator
from torchturbo.runtime.cpu import CpuRuntime


@pytest.mark.parametrize('tp_size', [1, 2, 4])
@pytest.mark.parametrize('checkpointing', [False, True])
def test_useful_share_and_only_legacy_hfu(monkeypatch, tp_size, checkpointing):
    notices = []
    monkeypatch.setattr("torchturbo.engine.metrics.mfu_estimator.get_logger", lambda name: SimpleNamespace(info_rank0=lambda *args: notices.append(args[0]), debug=lambda *args: None))
    clock = iter([0., 2., 4.])
    monkeypatch.setattr('torchturbo.engine.metrics.mfu_estimator.time.perf_counter', lambda: next(clock))
    class Recipe:
        def __init__(self, config, profile):
            pass
        def components(self):
            # Full head/router remain replicated: ideal hardware work is larger
            # than useful share. This test deliberately does not claim HFU.
            return [SimpleNamespace(fwd_flops=800), SimpleNamespace(fwd_flops=200)]
    config = EngineConfig(mfu=MfuConfig(enabled=True, card='h100'))
    config.activation_checkpointing.enabled = checkpointing
    estimator = MFUEstimator(Recipe, config, SimpleNamespace(), {'sp': {'sp_size': 2}, 'tp': {'tp_size': tp_size}}, CpuRuntime())
    inputs = {'input_ids': torch.ones(1, 8), 'cu_seq_lens_q': torch.tensor([0, 3, 8])}
    assert estimator.forward_flops(inputs) == 1000 // (2 * tp_size)
    estimator.record_micro(inputs)
    report = estimator.report()
    assert ('hfu' in report) == (tp_size == 1)
    assert report['mfu'].item() == pytest.approx(3 * 1000 / (2 * tp_size) / 2 / 989e12)
    assert estimator.report() == {}
    assert sum("TP HFU is unavailable" in msg for msg in notices) == int(tp_size > 1)


@pytest.mark.parametrize('drop_last,expected', [(True, 12), (False, 24)])
def test_literal_sampler_ids_shared_only_within_data_replica(monkeypatch, drop_last, expected):
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 12)
    lanes = []
    for rank in range(12):
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda: rank)
        engine = TorchTurboEngine.__new__(TorchTurboEngine)
        engine.config = SimpleNamespace(gbs=6)
        engine.get_dp_info = lambda: {'dp_size': 2, 'dp_rank': rank // 6}
        engine.get_sp_info = lambda: {'sp_size': 3}
        loader = engine.create_dataloader(list(range(13)), batch_size=2, shuffle=False, pin_memory=False, drop_last=drop_last)
        ids = [sample for micro in loader for sample in micro.tolist()]
        lanes.append(ids)
        assert loader.sampler.total_size == expected
    assert all(ids == lanes[0] for ids in lanes[:6])
    assert all(ids == lanes[6] for ids in lanes[6:])
    assert lanes[0][:6] == [0, 1, 2, 3, 4, 5]
    assert lanes[6][:6] == [6, 7, 8, 9, 10, 11]
