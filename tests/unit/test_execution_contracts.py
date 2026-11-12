from copy import deepcopy
import json
import struct

import pytest

from vibeforge.contracts.campaign import load_config
from vibeforge.contracts.precision import normalize_precision, precision_fields, validate_candidate_runtime
from vibeforge.controller.admission import decide
from vibeforge.evaluation.correctness import numeric_report
from vibeforge.profiling.collector import collect
from vibeforge.profiling.model import inspect_model
from vibeforge.profiling.sampler import estimate_memory, deployment_preflight
from vibeforge.storage.atomic import write_json


@pytest.mark.parametrize("quant,weight", [("none", "float32"), ("bnb-nf4", "int4"),
                                         ("bnb-fp4", "int4"), ("bnb-int8", "int8")])
def test_precision_profiles_are_explicit(quant, weight):
    precision = normalize_precision({"compute_dtype": "fp32", "quantization": quant})
    assert precision["weight_dtype"] == weight
    assert precision["compute_dtype"] == "float32"
    assert normalize_precision(precision) == precision


@pytest.mark.parametrize("value", [[], {"compute_dtype": "fp4"}, {"quantization": "awq"},
                                   {"quantization": "bnb-nf4", "weight_dtype": "int8"}])
def test_invalid_precision_is_rejected(value):
    with pytest.raises(ValueError):
        normalize_precision(value)


def test_precision_changes_need_authorization_and_allowlist(toy_config):
    parent = {"backend": "toy", **precision_fields(toy_config["precision"])}
    child = {**parent, **precision_fields(normalize_precision({"compute_dtype": "float32"}))}
    with pytest.raises(ValueError, match="not authorized"):
        validate_candidate_runtime(parent, child, toy_config)
    toy_config["optimization"]["allow_precision_changes"] = True
    with pytest.raises(ValueError, match="allowlist"):
        validate_candidate_runtime(parent, child, toy_config)
    toy_config["optimization"]["allowed_precisions"] = [child["precision"]]
    assert validate_candidate_runtime(parent, child, toy_config) == child["precision"]
    child["dtype"] = "float16"
    with pytest.raises(ValueError, match="disagrees"):
        validate_candidate_runtime(parent, child, toy_config)


def test_scheduler_changes_need_surface_authorization(toy_config):
    parent = {"backend": "toy", "request_scheduler": "serialized_lock",
              **precision_fields(toy_config["precision"])}
    child = {**parent, "request_scheduler": "concurrent_requests"}
    toy_config["optimization"]["allowed_surfaces"] = ["execution_adapter"]
    with pytest.raises(ValueError, match="scheduler change"):
        validate_candidate_runtime(parent, child, toy_config)
    toy_config["optimization"]["allowed_surfaces"].append("scheduler")
    assert validate_candidate_runtime(parent, child, toy_config) == child["precision"]
    child["request_scheduler"] = "continuous_batching"
    with pytest.raises(ValueError, match="serialized_lock, concurrent_requests or microbatch"):
        validate_candidate_runtime(parent, child, toy_config)


def test_approximation_requires_independent_quality_gate(project, tmp_path, toy_config):
    toy_config["evaluation"]["correctness_mode"] = "approximate"
    path = tmp_path / "campaign.json"
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="quality evaluator"):
        load_config(path, project)


def test_production_self_reference_requires_independent_quality(project, tmp_path):
    config = load_config(project / "configs/campaigns/qwen36-35b.json", project)
    config["evaluation"].pop("quality")
    config.pop("config_hash", None)
    path = tmp_path / "campaign.json"
    write_json(path, config)
    with pytest.raises(ValueError, match="self-generated regression reference"):
        load_config(path, project)


def test_memory_estimate_uses_target_dtype_and_preserves_integer_buffers():
    model = {"weight_bytes": 208, "dtype_elements": {"BF16": 100, "I64": 1}, "dtype_bytes": {"BF16": 200, "I64": 8}}
    a = estimate_memory(model, {"precision": {"compute_dtype": "bfloat16"}, "reserve_gib": 0})
    b = estimate_memory(model, {"precision": {"compute_dtype": "float32"}, "reserve_gib": 0})
    c = estimate_memory(model, {"precision": {"quantization": "bnb-int8"}, "reserve_gib": 0})
    assert a["runtime_weight_bytes_estimate"] == 208
    assert b["runtime_weight_bytes_estimate"] == 408
    assert c["runtime_weight_bytes_estimate"] == 138


def test_hardware_device_index_and_bf16_are_checked(monkeypatch):
    monkeypatch.setattr("vibeforge.profiling.sampler.memory_info", lambda: {"available_bytes": 10**12})
    monkeypatch.setattr("vibeforge.profiling.sampler.gpu_processes", lambda: [])
    environment = {"inference": {"packages": {}, "runtime": {"backend": "rocm", "devices": [
        {"index": 0, "free_bytes": 10**10, "bf16_supported": False}]}}}
    target = {"device": "rocm:0", "engine": {"required_packages": []}, "reserve_gib": 0}
    result = deployment_preflight({"weight_bytes": 100}, {}, target, environment)
    assert result["reasons"] == ["device_bfloat16_unsupported"]
    target["device"] = "rocm:1"
    assert deployment_preflight({"weight_bytes": 100}, {}, target, environment)["reasons"] == ["rocm_device_unavailable"]


def test_gguf_and_manifest_inputs_validate_declared_boundary(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
    profile = inspect_model(path, "gguf")
    assert profile["prequantized"] and "header" in profile["validation"]
    manifest = tmp_path / "model_manifest.json"
    write_json(manifest, {"artifacts": [{"path": "model.gguf"}], "model_type": "custom"})
    assert inspect_model(manifest, "manifest")["weight_bytes"] == 24
    write_json(manifest, {"artifacts": [{"path": "../outside"}]})
    with pytest.raises(ValueError, match="unconfined"):
        inspect_model(manifest, "manifest")


def test_profile_separates_precision_and_ignores_free_memory(monkeypatch, toy_config):
    environment = {"inference": {"runtime": {"devices": [{"free_bytes": 100, "index": 0}]}}}
    monkeypatch.setattr("vibeforge.profiling.collector.inspect_environment", lambda *args: deepcopy(environment))
    monkeypatch.setattr("vibeforge.profiling.collector.inspect_hardware", lambda: {"cpu_count": 8})
    first, _ = collect(toy_config)
    environment["inference"]["runtime"]["devices"][0]["free_bytes"] = 50
    second, _ = collect(toy_config)
    assert first["profile_hash"] == second["profile_hash"]
    toy_config["precision"] = normalize_precision({"compute_dtype": "float32"})
    third, _ = collect(toy_config)
    assert first["profile_hash"] != third["profile_hash"]


def test_latency_objective_and_resource_guard_are_independent():
    def row(latency, rss=100):
        return {"valid": True, "requests": [{} for _ in range(100)], "metrics": {"latency_p95_seconds": latency, "peak_rss_bytes": rss}}
    protocol = {"objective": "latency_p95_seconds", "min_gain": .02, "limits": {"peak_rss_bytes": 200}}
    baseline, candidate = [row(2)] * 5, [row(1)] * 5
    result = decide({"passed": True}, baseline, candidate, protocol)
    assert result.outcome == "accept" and result.gain == .5 and result.direction == "minimize"
    result = decide({"passed": True}, baseline, [row(1, 201)] * 5, protocol)
    assert result.reason == "protection_failed:peak_rss_bytes"


def test_normalized_error_cannot_be_hidden_by_large_absolute_tolerance():
    result = numeric_report({"indices": [0, 1], "values": [1, 2]}, {"indices": [0, 1], "values": [1, 1]},
                            {"atol": 10, "rtol": 10, "max_normalized_rms": .1})
    assert not result["passed"]
