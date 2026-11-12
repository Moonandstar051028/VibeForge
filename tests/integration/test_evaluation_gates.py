import importlib.util
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from vibeforge.evaluation.correctness import compare_generation, evaluate_correctness, numeric_report
from vibeforge.evaluation.performance import measure
from vibeforge.evaluation.quality import run_quality
from vibeforge.evaluation.reference import capture_reference, validate_reference
from vibeforge.engine.process import EngineProcess
from vibeforge.storage.atomic import digest


def test_fixed_length_fast_path_is_checked_in_correctness_and_benchmark(project, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fixture", project / "templates/engine/toy/engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load({})
    workload = {"prompts": ["hello"], "max_new_tokens": 4, "concurrency": 2}
    reference = capture_reference(module, [{"id": "normal", "prompt": "normal", "max_new_tokens": 2}],
                                  tmp_path / "reference.json", {"fixture": True}, {}, workload)
    original = module.generate
    def wrong_fast_path(**kwargs):
        value = original(**kwargs)
        if kwargs.get("fixed_length"):
            value["token_ids"] = [0] * len(value["token_ids"])
        return value
    module.generate = wrong_fast_path
    result = evaluate_correctness(module, reference, {})
    assert not result["passed"]
    assert next(c for c in result["checks"] if c["case"] == "normal")["passed"]
    monkeypatch.setattr("vibeforge.evaluation.performance.sample", lambda: {"gpu_processes": []})
    result = measure(SimpleNamespace(generate=lambda **kw: module.generate(**{k: v for k, v in kw.items() if k != "observation_mode"})), workload, tmp_path / "samples.jsonl", "test", 0, {}, reference=reference)
    assert not result["valid"] and "benchmark_output_incorrect" in result["invalid_reasons"]


def test_performance_uses_frozen_device_and_tail_sampling_protocol(project, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fixture", project / "templates/engine/toy/engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load({})
    workload = {"prompts": ["hello"], "max_new_tokens": 4, "concurrency": 1}
    reference = capture_reference(
        module, [{"id": "normal", "prompt": "normal", "max_new_tokens": 2}],
        tmp_path / "reference.json", {"fixture": True}, {}, workload)
    observed = {}

    class DeviceSampler:
        def __init__(self, interval, enabled):
            observed.update(interval=interval, enabled=enabled)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def report(self):
            return {
                "status": "collected", "devices": [{"metrics": {
                    "temperature_c": {"maximum": 90.0},
                }}],
            }

    monkeypatch.setattr("vibeforge.evaluation.performance.DeviceTelemetrySampler", DeviceSampler)
    monkeypatch.setattr("vibeforge.evaluation.performance.sample", lambda: {
        "gpu_processes": [], "gpu_temperature_c": 30.0,
    })
    protocol = {
        "device_backend": "cuda", "device_sample_interval": 0.25,
        "require_continuous_device_telemetry": True,
        "max_temperature_c": 80, "min_tail_requests": 20,
    }

    result = measure(
        SimpleNamespace(generate=lambda **kw: module.generate(**{k: v for k, v in kw.items() if k != "observation_mode"})), workload, tmp_path / "samples.jsonl", "test", 0, protocol,
        reference=reference)

    assert observed == {"interval": 0.25, "enabled": True}
    assert "temperature_limit" in result["invalid_reasons"]
    assert result["tail_latency"] == {
        "request_count": 1, "minimum_reliable_requests": 20,
        "sample_size_sufficient": False,
        "scope": "request quantiles within this profile and measurement group",
    }


def test_quality_input_mutation_is_detected(tmp_path):
    script = tmp_path / "quality.py"
    script.write_text("original")
    settings = {"command": [sys.executable, str(script)], "file_hashes": {str(script): "wrong"}}
    class Client:
        url = "http://127.0.0.1:1"
    with pytest.raises(ValueError, match="modified"):
        run_quality(Client(), settings, tmp_path / "results", {"project": str(tmp_path)})


def test_quality_process_failure_cannot_reuse_stale_success(tmp_path):
    class Client:
        url = "http://127.0.0.1:1"
    from vibeforge.storage.atomic import write_json
    folder = tmp_path / "quality"
    write_json(folder / "result.json", {"passed": True, "metrics": {"accuracy": 1}})
    settings = {"command": [sys.executable, "-c", "raise SystemExit(7)"],
                "timeout_seconds": 2, "min_scores": {"accuracy": .9}}
    report = run_quality(Client(), settings, folder, {"project": str(tmp_path)})
    assert not report["passed"] and "7" in report["errors"][0]
    assert list(folder.glob("stale-result-*.json"))


@pytest.mark.parametrize("invalid_metrics", ["None", "[]", "[1]", "'invalid'"])
def test_malformed_quality_metrics_fail_repeated_gate_without_losing_report(tmp_path, invalid_metrics):
    class Client:
        url = "http://127.0.0.1:1"
    code = ("import json,sys; "
            "json.dump(dict(passed=True, metrics=" + invalid_metrics + "), open(sys.argv[1], 'w'))")
    settings = {"command": [sys.executable, "-c", code, "{output}"],
                "repeats": 2, "timeout_seconds": 2, "min_scores": {"accuracy": .9}}
    report = run_quality(Client(), settings, tmp_path / "quality", {"project": str(tmp_path)})
    assert not report["passed"] and report["metrics"] == {}
    assert len(report["runs"]) == 2
    assert all("Quality result metrics must be an object" in error for error in report["errors"])
    assert (tmp_path / "quality/report.json").exists()


def test_supplied_task_quality_evaluator_calls_engine_and_scores_cases(paths, project, tmp_path):
    from vibeforge.storage.atomic import write_json
    cases = tmp_path / "cases.json"
    write_json(cases, [
        {"id": "generate", "mode": "generate", "prompt": "hello", "max_new_tokens": 2,
         "expected": r"^\d+ \d+$", "match": "regex"},
        {"id": "chat", "mode": "chat", "prompt": "ready?", "expected": "toy protocol ready", "match": "contains"},
        {"id": "wrong", "mode": "generate", "prompt": "hello", "max_new_tokens": 2,
         "expected": "wrong answer", "match": "exact"},
    ])
    engine = EngineProcess(paths, "task-quality", sys.executable, startup_timeout=10)
    try:
        client = engine.start(project / "templates/engine/toy", tmp_path / "engine.log")
        settings = {"command": ["{python}", "{project}/scripts/evaluation/quality_cases.py", "--request", "{request}",
                               "--output", "{output}", "--cases", str(cases)],
                    "min_scores": {"accuracy": .6}, "timeout_seconds": 10}
        report = run_quality(client, settings, tmp_path / "quality", {"project": str(project)})
        assert report["passed"] and report["metrics"]["accuracy"] == 2 / 3
    finally:
        engine.stop()


def test_task_quality_request_failure_invalidates_an_otherwise_high_score(project, monkeypatch):
    import io
    spec = importlib.util.spec_from_file_location("quality_cases", project / "scripts/evaluation/quality_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = 0
    def response(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 10:
            raise OSError("engine disconnected")
        return io.StringIO('{"text": "4"}')
    monkeypatch.setattr(module, "urlopen", response)
    cases = [{"id": str(index), "prompt": "2 + 2", "expected": "4"} for index in range(10)]
    result = module.evaluate("http://127.0.0.1:1", cases, 1)
    assert result["metrics"]["accuracy"] == .9
    assert not result["passed"]
    assert result["cases"][-1]["error"] == "engine disconnected"


def test_repeated_reference_records_generation_variants_and_logit_envelope(tmp_path):
    class VariableClient:
        def __init__(self):
            self.generations = 0
            self.numeric = 0

        def generate(self, **kwargs):
            self.generations += 1
            token = 2 if self.generations < 3 else 3
            return {"input_ids": [1], "token_ids": [token], "text": str(token)}

        def logits(self, **kwargs):
            value = [1.0, 1.00005, 1.0001][self.numeric % 3]
            self.numeric += 1
            return {"indices": kwargs["indices"], "values": [value]}

    reference = capture_reference(
        VariableClient(), [{"id": "variable", "prompt": "x", "max_new_tokens": 1}],
        tmp_path / "reference.json", {"vocab_size": 1},
        {"reference_repeats": 3, "min_reference_consensus": 2 / 3},
    )
    case = reference["cases"][0]
    assert reference["schema_version"] == 3 and case["reference_consensus"] == pytest.approx(2 / 3)
    assert case["accepted_token_ids"] == [[2], [3]]
    assert compare_generation({"input_ids": [1], "token_ids": [3]}, case, {})
    protocol = {"atol": 0, "rtol": 0, "reference_atol": .0001}
    assert numeric_report({"indices": [0], "values": [1.000075]}, case["logits"], protocol)["passed"]
    assert not numeric_report({"indices": [0], "values": [1.001]}, case["logits"], protocol)["passed"]
    assert not compare_generation({"input_ids": [1], "token_ids": [4]}, case, {})
    invalid = deepcopy(reference)
    invalid["cases"][0]["output_variants"][0]["count"] = 10
    with pytest.raises(ValueError, match="counts or consensus"):
        validate_reference(invalid)
    with pytest.raises(ValueError, match="does not cover configured case"):
        validate_reference(reference, cases=[{"id": "missing", "prompt": "x"}])


def test_strict_reference_rejects_an_unstable_initial_engine(tmp_path):
    class UnstableClient:
        def __init__(self):
            self.call = 0

        def generate(self, **kwargs):
            self.call += 1
            return {"input_ids": [1], "token_ids": [self.call], "text": str(self.call)}

        def logits(self, **kwargs):
            return {"indices": kwargs["indices"], "values": [1.0]}

    with pytest.raises(ValueError, match="Reference generation is unstable"):
        capture_reference(
            UnstableClient(), [{"id": "unstable", "prompt": "x", "max_new_tokens": 1}],
            tmp_path / "reference.json", {"vocab_size": 1},
            {"reference_repeats": 3, "min_reference_consensus": 2 / 3},
        )


def test_quality_repeats_are_aggregated_conservatively(tmp_path):
    class Client:
        url = "http://127.0.0.1:1"

    script = ("import json; from pathlib import Path; "
              "score = 0.7 if Path.cwd().name == '0002' else 1.0; "
              "json.dump(dict(passed=True, metrics=dict(accuracy=score)), open(r'{output}', 'w'))")
    settings = {"command": [sys.executable, "-c", script], "repeats": 3,
                "timeout_seconds": 5, "min_scores": {"accuracy": .6}, "max_regressions": {"accuracy": .1}}
    report = run_quality(Client(), settings, tmp_path / "quality", {"project": str(tmp_path)},
                         baseline={"metrics": {"accuracy": 1.0}})
    assert not report["passed"] and report["repeat_count"] == 3
    assert report["metrics"] == {"accuracy": .7}
    assert report["metric_statistics"]["accuracy"]["mean"] == pytest.approx(.9)
    assert "run_2:quality_regression:accuracy" in report["errors"]
    assert len(report["runs"]) == 3


@pytest.mark.parametrize("values", [[1.0, 100.0, 1.0], [float("nan")] * 3])
def test_numeric_instability_cannot_expand_the_reference(tmp_path, values):
    class Client:
        call = 0

        def generate(self, **kwargs):
            return {"input_ids": [1], "token_ids": [2]}

        def logits(self, indices, **kwargs):
            value = values[self.call % 3]
            self.call += 1
            return {"indices": indices, "values": [value]}

    with pytest.raises(ValueError, match="Reference"):
        capture_reference(Client(), [{"id": "x", "prompt": "x", "max_new_tokens": 1}],
                          tmp_path / "ref.json", {"vocab_size": 1}, {"atol": .01, "rtol": .01})
    assert not (tmp_path / "ref.json").exists()


@pytest.mark.parametrize("minimum,maximum", [([], [1]), ([2], [1]), ([float("nan")], [1])])
def test_malformed_numeric_envelope_fails_closed(minimum, maximum):
    report = numeric_report({"indices": [0], "values": [1]},
                            {"indices": [0], "values": [1], "min_values": minimum, "max_values": maximum}, {})
    assert not report["passed"] and report["error"] == "invalid_reference_envelope"
