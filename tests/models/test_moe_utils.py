"""CPU split metadata consumed by the real all-to-all dispatcher."""

from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchturbo.models.modules.comm import all_to_all
from torchturbo.models.modules.moe.utils import maybe_move_tensor_to_cpu


def test_cpu_split_metadata_numpy_conversion():
    splits = torch.tensor([1, 2], dtype=torch.int64)
    converted = maybe_move_tensor_to_cpu(splits, as_numpy=True)
    assert isinstance(converted, np.ndarray)
    np.testing.assert_array_equal(converted, [1, 2])
    assert maybe_move_tensor_to_cpu(splits) is splits
    for value in (None, 3, [1, 2], np.array([1, 2])):
        assert maybe_move_tensor_to_cpu(value, as_numpy=True) is value


def _all_to_all_worker(rank, path):
    dist.init_process_group('gloo', init_method=f'file://{path}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        send = maybe_move_tensor_to_cpu(torch.tensor([[1, 2], [2, 1]][rank]), as_numpy=True)
        values = (torch.arange(3.).view(3, 1) + rank * 10).requires_grad_()
        output = all_to_all(dist.group.WORLD, values, send, send)
        expected = [[0., 10., 11.], [1., 2., 12.]][rank]
        torch.testing.assert_close(output[:, 0], torch.tensor(expected))
        (output * (rank + 1)).sum().backward()
        torch.testing.assert_close(values.grad[:, 0], torch.tensor([[1., 2., 2.], [1., 1., 2.]][rank]))
    finally:
        dist.destroy_process_group()


def test_cpu_all_to_all_numpy_splits_forward_backward(tmp_path):
    mp.spawn(_all_to_all_worker, args=(str(tmp_path / 'store'),), nprocs=2, join=True)
