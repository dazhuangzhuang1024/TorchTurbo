"""Actual native export CLI under the explicit CPU construction bridge."""

import sys
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.checkpoint.dcp_to_hf import main
from .etp_helpers import cpu_bridge


def worker(rank, world, rendezvous, argv):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
        world_size=world, timeout=timedelta(seconds=120))
    with cpu_bridge():
        main(argv)
    # The actual CLI owns and destroys its process group.
    Path(rendezvous).parent.joinpath(f"cli-rank{rank}.json").write_text('{"completed":true}')


if __name__ == "__main__":
    mp.spawn(worker, args=(int(sys.argv[1]), sys.argv[2], sys.argv[3:]), nprocs=int(sys.argv[1]), join=True)
