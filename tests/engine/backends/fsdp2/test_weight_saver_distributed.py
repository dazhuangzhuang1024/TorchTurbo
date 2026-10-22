"""Two-rank HF writer ordering and failure propagation tests."""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

from torchturbo.engine.backends.fsdp2.checkpoint import weight_saver
from torchturbo.engine.checkpoint.distributed import collective_check


_HAS_GLOO = dist.is_available() and dist.is_gloo_available()


class _ProcessingClass:
    def __init__(self, fail: bool) -> None:
        self.fail = fail

    def save_pretrained(self, path: Path) -> None:
        if self.fail:
            raise OSError("rank-zero metadata finalization failure")
        (path / "tokenizer_config.json").write_text('{"tokenizer_class": "Example"}')


def _run_hf_writer_test(
    rank: int,
    init_file: str,
    directory: str,
    mode: str,
    temp_ready,
    rename_allowed,
    renamed,
    rank_one_write_complete,
    metadata_started_early,
) -> None:
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        root = Path(directory)
        export = root / "export"
        name = f"weight_{rank}"
        saver = weight_saver.ShardedParamManager.__new__(weight_saver.ShardedParamManager)
        saver.rank = rank
        saver.world_size = 2
        saver.sharded_param_groups = [[name]]
        saver.local_params = {name}
        saver.src_path = root / "source"
        saver.processing_class = _ProcessingClass(mode == "metadata_failure")
        saver.model = SimpleNamespace(
            config=SimpleNamespace(to_dict=lambda: {"num_nextn_predict_layers": 0})
        )
        saver._src_metadata = {"total_size": 16, "format": "pt"}
        saver._get_local_hf_state_dict = lambda local_params: {
            name: torch.tensor([float(rank), float(rank + 10)])
        }
        original_save_file = weight_saver.save_file
        original_walk = os.walk
        original_copy_metadata = saver._copy_non_safetensor_files

        def controlled_save_file(shard, file_path):
            if rank == 1 and mode == "shard_failure":
                raise OSError("rank-one shard write failure")
            if mode != "rename_race":
                original_save_file(shard, file_path)
                return
            if rank == 0:
                original_save_file(shard, file_path)
                assert temp_ready.wait(20), "rank one did not write its temporary shard"
                return

            # Model the observed atomic-write race with a real safetensors
            # payload. This does not characterize the Rust writer's internals.
            temp_path = export / ".tmp-rank-one-shard"
            original_save_file(shard, str(temp_path))
            temp_ready.set()
            assert rename_allowed.wait(20), "rank zero did not release the rename"
            os.replace(temp_path, file_path)
            renamed.set()

        def controlled_walk(path):
            for directory_path, directories, files in original_walk(path):
                if rank == 0 and ".tmp-rank-one-shard" in files:
                    # Hold the old listing until the temporary path disappears.
                    rename_allowed.set()
                    assert renamed.wait(20), "rank one did not rename its shard"
                yield directory_path, directories, files

        def release_after_local_action(action):
            def local_action():
                result = action()
                if rank == 0:
                    # Correct ordering ends the local action before scanning.
                    # The old ordering releases the rename through os.walk.
                    rename_allowed.set()
                else:
                    rank_one_write_complete.set()
                return result

            return collective_check(local_action)

        def track_metadata_start(source, destination):
            if not rank_one_write_complete.is_set():
                metadata_started_early.set()
            original_copy_metadata(source, destination)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(weight_saver, "save_file", controlled_save_file)
            if mode == "rename_race":
                patch.setattr(os, "walk", controlled_walk)
                patch.setattr(weight_saver, "collective_check", release_after_local_action)
                patch.setattr(saver, "_copy_non_safetensor_files", track_metadata_start)
            try:
                saver.save_model(export)
            except Exception as exc:
                outcome = (type(exc).__name__, str(exc))
            else:
                outcome = ("success", "")

        outcomes = [None, None]
        dist.all_gather_object(outcomes, outcome)
        index_path = export / weight_saver.HF_INDEX_FILE
        if mode == "rename_race":
            assert outcomes == [("success", ""), ("success", "")], outcomes
            assert renamed.is_set()
            assert not metadata_started_early.is_set()
            index = json.loads(index_path.read_text())
            assert index == {
                "metadata": {"total_size": 16, "format": "pt"},
                "weight_map": {
                    "weight_0": "model-00000-of-00002.safetensors",
                    "weight_1": "model-00001-of-00002.safetensors",
                },
            }
            for weight, filename, expected in (
                ("weight_0", "model-00000-of-00002.safetensors", [0.0, 10.0]),
                ("weight_1", "model-00001-of-00002.safetensors", [1.0, 11.0]),
            ):
                saved = load_file(str(export / filename))
                assert set(saved) == {weight}
                torch.testing.assert_close(saved[weight], torch.tensor(expected), rtol=0, atol=0)
            assert json.loads((export / "config.json").read_text()) == {
                "num_nextn_predict_layers": 0,
                "use_cache": True,
            }
            assert json.loads((export / "tokenizer_config.json").read_text()) == {
                "tokenizer_class": "Example"
            }
            assert (export / "assets" / "vocab.txt").read_text() == "example\n"
            assert not (export / ".tmp-rank-one-shard").exists()
        else:
            message = (
                "rank-one shard write failure"
                if mode == "shard_failure"
                else "rank-zero metadata finalization failure"
            )
            assert outcomes == [("OSError", message), ("OSError", message)], outcomes
            assert not index_path.exists()
            if mode == "shard_failure":
                assert not (export / "config.json").exists()
    finally:
        dist.destroy_process_group()


def _spawn_hf_writer_test(tmp_path: Path, mode: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        '{"num_nextn_predict_layers": 2, "use_cache": true}'
    )
    (source / "assets").mkdir()
    (source / "assets" / "vocab.txt").write_text("example\n")
    context = mp.get_context("spawn")
    events = [context.Event() for _ in range(5)]
    mp.spawn(
        _run_hf_writer_test,
        args=(str(tmp_path / "init"), str(tmp_path), mode, *events),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(not _HAS_GLOO, reason="requires torch.distributed with Gloo support")
def test_hf_metadata_scan_waits_for_parallel_shard_renames(tmp_path):
    _spawn_hf_writer_test(tmp_path, "rename_race")


@pytest.mark.skipif(not _HAS_GLOO, reason="requires torch.distributed with Gloo support")
@pytest.mark.parametrize("mode", ["shard_failure", "metadata_failure"])
def test_hf_export_failures_reach_every_rank_without_publishing_index(tmp_path, mode):
    _spawn_hf_writer_test(tmp_path, mode)
