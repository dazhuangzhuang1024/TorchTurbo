"""Exact HF conversion at the packed/logical grouped-expert boundary."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext
from tests.engine.backends.fsdp2.etp_checkpoint_helpers import (
    CheckpointMoE, expert_values, hf_values, write_source,
)


def _converter_worker(rank, directory, logical, format):
    dist.init_process_group("gloo", init_method=f"file://{directory}/store", rank=rank,
                            world_size=4, timeout=timedelta(seconds=90))
    try:
        context = ParallelContext("cpu", sp_size=1, ep_size=2,
                                  tp_size=2 if logical else 1, etp_size=2 if logical else 1)
        model = CheckpointMoE(context, 2, fused=format != "base")
        expected = hf_values(model, 4, fused=format == "fused")
        source = Path(directory, "source")
        if rank == 0:
            write_source(source, expected)
        dist.barrier()
        converter = model.get_weight_converter({"ep": context.get_ep_info(),
                                               "etp": context.get_etp_info()}, str(source))
        weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        full_w1, full_w2 = expert_values(4)
        start = context.ep_mesh.get_local_rank() * 2
        for index, oracle in ((1, full_w1[start:start + 2].reshape(6, 2, 4)),
                              (2, full_w2[start:start + 2])):
            name = f"moe.experts.weight{index}"
            entries = converter.get_expert_load_entries(name, weight_map)
            assert converter.get_expert_load_entries(name, weight_map) == entries
            _, (_, local_names, _), _ = entries
            local_sd = {}
            for key in local_names:
                local_sd[key] = (converter.slice_virtual_key(key, expected[converter.get_physical_key(key)])
                                 if converter.is_virtual_key(key) else expected[key].clone())
            target = torch.empty_like(oracle if logical else oracle.flatten(1) if index == 1 else oracle.flatten(0, 1))
            converter.assemble_expert_weight_for_broadcast(target, local_sd, name)
            assert not local_sd
            torch.testing.assert_close(target.reshape_as(oracle), oracle, rtol=0, atol=0)
            export = (converter.get_local_ep_w1_hf_state_dict if index == 1
                      else converter.get_local_ep_w2_hf_state_dict)(name, target)
            assert export
            for key, value in export.items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            if logical:
                wrong = target.reshape(-1)
                with pytest.raises(ValueError, match="logical.*shape|shape.*logical"):
                    (converter.get_local_ep_w1_hf_state_dict if index == 1
                     else converter.get_local_ep_w2_hf_state_dict)(name, wrong)
                with pytest.raises(ValueError, match="logical.*shape|shape.*logical"):
                    converter.assemble_expert_weight_for_broadcast(wrong, {}, name)
        Path(directory, f"rank-{rank}.json").write_text(json.dumps({"logical": logical, "format": format, "ep": start // 2}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("logical", [False, True], ids=["packed", "logical"])
@pytest.mark.parametrize("format", ["base", "fused", "fallback"])
def test_logical_expert_converter_roundtrip(tmp_path, logical, format):
    mp.spawn(_converter_worker, args=(str(tmp_path), logical, format), nprocs=4, join=True)
    assert len(list(tmp_path.glob("rank-*.json"))) == 4
