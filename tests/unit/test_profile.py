import json
import struct
import pytest
from vibeforge.profiling.model import inspect_model
from vibeforge.profiling.sampler import deployment_preflight
from vibeforge.storage.atomic import write_json


def tiny_checkpoint(tmp_path):
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)
    write_json(tmp_path / "config.json", {"model_type": "fixture"})
    write_json(tmp_path / "model.safetensors.index.json",
               {"metadata": {"total_size": 4}, "weight_map": {"weight": "model.safetensors"}})
    for name in ("tokenizer_config.json", "tokenizer.json"):
        write_json(tmp_path / name, {})
    return tmp_path


def test_header_inspection_checks_actual_bounds(tmp_path):
    root = tiny_checkpoint(tmp_path)
    assert inspect_model(root)["tensor_count"] == 1
    shard = root / "model.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="bounds"):
        inspect_model(root)


def test_preflight_blocks_memory_shortfall_without_allocating(monkeypatch):
    monkeypatch.setattr("vibeforge.profiling.sampler.memory_info", lambda: {"available_bytes": 1024})
    monkeypatch.setattr("vibeforge.profiling.sampler.gpu_processes", lambda: [])
    result = deployment_preflight({"weight_bytes": 10**9}, {"gpu": "test"}, {"reserve_gib": 1},
                                  {"inference": {"packages": dict.fromkeys(
                                      ["torch", "transformers", "safetensors", "accelerate"], "installed")}})
    assert not result["passed"]
    assert result["reasons"] == ["insufficient_memory"]
