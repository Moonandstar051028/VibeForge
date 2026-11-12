import json
import struct
from types import SimpleNamespace

import pytest
from vibeforge.engine.observation import _tensor_signatures
from vibeforge.profiling.model import inspect_model
from vibeforge.profiling.sampler import deployment_preflight
from vibeforge.storage.atomic import write_json


def test_tensor_signatures_are_bounded_deduplicated_metadata():
    first = SimpleNamespace(shape=(2, 16, 64), dtype="bf16", device="cuda:0")
    duplicate = SimpleNamespace(shape=(2, 16, 64), dtype="bf16", device="cuda:0")
    second = SimpleNamespace(shape=(2, 1, 64), dtype="bf16", device="cuda:0")

    signatures = _tensor_signatures({"hidden": [first, duplicate], "decode": second})

    assert signatures == [
        {"shape": [2, 16, 64], "dtype": "bf16", "device": "cuda:0"},
        {"shape": [2, 1, 64], "dtype": "bf16", "device": "cuda:0"},
    ]


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
    monkeypatch.setattr("vibeforge.profiling.sampler.visible_gpus", lambda: [
        {"uuid": "GPU-test", "available_bytes": 8 * 1024 ** 3}])
    result = deployment_preflight({"weight_bytes": 10**9}, {"gpu": "test"}, {"reserve_gib": 1},
                                  {"inference": {"packages": dict.fromkeys(
                                      ["torch", "transformers", "safetensors", "accelerate"], "installed")}})
    assert not result["passed"]
    assert result["reasons"] == ["insufficient_memory"]


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_preflight_checks_gpu_memory_separately_from_host_memory(monkeypatch, device):
    gib = 1024 ** 3
    monkeypatch.setattr("vibeforge.profiling.sampler.memory_info", lambda: {"available_bytes": 400 * gib})
    monkeypatch.setattr("vibeforge.profiling.sampler.gpu_processes", lambda: [])
    monkeypatch.setattr("vibeforge.profiling.sampler.visible_gpus", lambda: [
        {"uuid": "GPU-test", "available_bytes": 40 * gib}])
    environment = {"inference": {"packages": dict.fromkeys(
        ["torch", "transformers", "safetensors", "accelerate"], "installed")}}
    result = deployment_preflight({"weight_bytes": 67 * gib}, {"gpu": "A100"},
                                  {"device": device}, environment)
    assert result["reasons"] == ["insufficient_gpu_memory"]
    assert result["gpu_memory"]["uuid"] == "GPU-test"
    result = deployment_preflight({"weight_bytes": 67 * gib}, {"gpu": "A100"},
                                  {"device": "cpu"}, environment)
    assert result["passed"] and result["gpu_memory"] is None


def test_visibility_orders_memory_and_excludes_other_device_processes(monkeypatch):
    from vibeforge.profiling.hardware import visible_gpus, gpu_processes

    def query(command):
        if command[1].startswith("--query-gpu="):
            return "0, GPU-zero, 40960, 40000\n1, GPU-one, 40960, 5000"
        return "101, 300, GPU-zero\n202, 35000, GPU-one"

    monkeypatch.setattr("vibeforge.profiling.hardware.query", query)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert [gpu["uuid"] for gpu in visible_gpus()] == ["GPU-zero"]
    assert gpu_processes() == [{"pid": 101, "memory_mib": "300"}]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-one,GPU-zero")
    assert [gpu["uuid"] for gpu in visible_gpus()] == ["GPU-one", "GPU-zero"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert visible_gpus() == [] and gpu_processes() == []


def test_gpu_telemetry_has_units_and_respects_visible_device(monkeypatch):
    from vibeforge.profiling.hardware import gpu_telemetry
    monkeypatch.setattr("vibeforge.profiling.hardware.visible_gpus", lambda: [{"uuid": "GPU-one"}])
    monkeypatch.setattr("vibeforge.profiling.hardware.query", lambda command:
        "0, GPU-zero, 80, 100, 50, 1410, 1215, 250, 250\n1, GPU-one, 30, 20, 10, 765, 1215, N/A, 250")
    rows = gpu_telemetry()
    assert len(rows) == 1 and rows[0]["visible_index"] == 0 and rows[0]["physical_index"] == 1
    assert rows[0]["temperature_c"] == 30 and rows[0]["power_watts"] is None


def test_device_telemetry_sampler_aggregates_each_visible_device(monkeypatch):
    from vibeforge.profiling.resources import DeviceTelemetrySampler

    rows = iter([
        [{"visible_index": 0, "physical_index": 1, "uuid": "GPU-one",
          "temperature_c": 30, "gpu_utilization_pct": 20}],
        [{"visible_index": 0, "physical_index": 1, "uuid": "GPU-one",
          "temperature_c": 40, "gpu_utilization_pct": 80}],
    ])
    monkeypatch.setattr("vibeforge.profiling.resources.gpu_telemetry", lambda: next(rows))
    sampler = DeviceTelemetrySampler(interval=1)
    sampler._sample()
    sampler._sample()

    report = sampler.report()

    assert report["status"] == "collected"
    assert report["sample_rounds"] == report["available_rounds"] == 2
    assert report["devices"][0]["metrics"]["temperature_c"] == {
        "samples": 2, "mean": 35.0, "minimum": 30.0, "maximum": 40.0,
    }
    assert report["devices"][0]["metrics"]["gpu_utilization_pct"]["mean"] == 50.0


def test_device_telemetry_sampler_reports_unavailable_without_fabricating_zero(monkeypatch):
    from vibeforge.profiling.resources import DeviceTelemetrySampler

    monkeypatch.setattr("vibeforge.profiling.resources.gpu_telemetry", lambda: None)
    sampler = DeviceTelemetrySampler(interval=1)
    sampler._sample()

    assert sampler.report()["status"] == "unavailable"
