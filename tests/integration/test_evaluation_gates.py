import importlib.util
import sys
from copy import deepcopy

import pytest

from vibeforge.evaluation.correctness import evaluate_correctness
from vibeforge.evaluation.performance import measure
from vibeforge.evaluation.quality import run_quality
from vibeforge.evaluation.reference import capture_reference
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
    result = measure(module, workload, tmp_path / "samples.jsonl", "test", 0, {}, reference=reference)
    assert not result["valid"] and "benchmark_output_incorrect" in result["invalid_reasons"]


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
