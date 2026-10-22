"""Run the actual export CLI with only the explicit CPU construction bridge."""

import sys
import json
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.checkpoint.dcp_to_hf import main
from tests.models.qwen3_moe.etp_helpers import cpu_qwen_bridge


def worker(rank, world, rendezvous, argv):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
        world_size=world, timeout=timedelta(seconds=90))
    with cpu_qwen_bridge():
        main(argv)
    Path(rendezvous).parent.joinpath(f"cli-rank{rank}.json").write_text(json.dumps({
        "rank": rank, "world": world, "completed": True,
    }))


if __name__ == "__main__":
    mp.spawn(worker, args=(int(sys.argv[1]), sys.argv[2], sys.argv[3:]),
             nprocs=int(sys.argv[1]), join=True)
