from types import SimpleNamespace
import pytest

from vibeforge.contracts.workload import normalize_workload, benchmark_cases, profiling_cases
from vibeforge.controller.admission import decide
from vibeforge.engine.observation import observe_call
from vibeforge.engine.observation import _torch_summaries
from vibeforge.engine.telemetry import _tensor_bytes
from vibeforge.evaluation.reference import capture_reference
from vibeforge.evaluation.performance import diagnose_concurrency, measure_suite


def test_profile_contract_validates_lengths_weights_and_ids(tmp_path):
    workload = {"profiles": [{"id": "short", "prompts": ["hello"], "max_new_tokens": 4},
                             {"id": "long", "prompts": ["hello world"], "max_new_tokens": 8, "concurrency": 2}]}
    normalize_workload(workload)
    assert [(c["profile_id"], c["max_new_tokens"]) for c in benchmark_cases(workload)] == [("short", 4), ("long", 8)]
    workload["profiles"][1]["id"] = "short"
    with pytest.raises(ValueError, match="unique"):
        normalize_workload(workload)


def test_profile_matrix_covers_profiles_before_additional_prompts():
    workload = normalize_workload({"observation_profile": "b", "profiles": [
        {"id": "a", "prompts": ["a0", "a1"], "max_new_tokens": 4},
        {"id": "b", "prompts": ["b0", "b1"], "max_new_tokens": 8},
        {"id": "c", "prompts": ["c0"], "max_new_tokens": 8},
    ]})

    all_cases = profiling_cases(workload, "all", 4)
    selected_cases = profiling_cases(workload, "selected", 8)

    assert [(row["profile"]["id"], row["prompt_index"]) for row in all_cases] == [
        ("a", 0), ("b", 0), ("c", 0), ("a", 1),
    ]
    assert [(row["profile"]["id"], row["prompt_index"]) for row in selected_cases] == [
        ("b", 0), ("b", 1),
    ]


def test_runtime_without_model_hooks_does_not_trust_self_reported_ttft():
    backend = SimpleNamespace(generate=lambda **kw: {"input_ids": [1], "token_ids": [2], "ttft_seconds": .0001})
    result, timing = observe_call(backend, {"device": "cpu"}, "generate", {})
    assert result["ttft_seconds"] == .0001 and timing["ttft_seconds"] is None
    assert timing["client_ttft_seconds"] is None and not timing["token_timing_available"]


def test_tokenizer_proxy_is_restored_after_failure():
    tokenizer = lambda text: text
    backend = SimpleNamespace(TOKENIZER=tokenizer)
    def fail():
        assert backend.TOKENIZER is not tokenizer
        raise ValueError("failure")
    backend.generate = fail
    with pytest.raises(ValueError, match="failure"):
        observe_call(backend, {"device": "cpu"}, "generate", {})
    assert backend.TOKENIZER is tokenizer


def test_kv_cache_logical_bytes_support_legacy_and_cache_objects():
    class Tensor:
        def __init__(self, elements, width):
            self.elements, self.width = elements, width

        def numel(self):
            return self.elements

        def element_size(self):
            return self.width

    legacy = ((Tensor(8, 2), Tensor(8, 2)),)
    cache = SimpleNamespace(key_cache=[Tensor(4, 4)], value_cache=[Tensor(4, 4)])

    assert _tensor_bytes(legacy) == 32
    assert _tensor_bytes(cache) == 32


def test_long_input_label_is_checked_against_actual_tokenizer_output(tmp_path):
    client = SimpleNamespace(generate=lambda **kw: {"input_ids": [1], "token_ids": [2]})
    workload = {"profiles": [{"id": "long", "prompts": ["not actually long"], "max_new_tokens": 1, "min_input_tokens": 100}]}
    with pytest.raises(ValueError, match="Actual tokenized"):
        capture_reference(client, [{"id": "long", "prompt": "x", "min_input_tokens": 100}], tmp_path / "ref.json", {}, {}, workload)


def test_suite_aggregation_keeps_profiles_and_geometric_weights(monkeypatch, tmp_path):
    import vibeforge.evaluation.performance as performance
    def fake(client, profile, *args, **kwargs):
        value = 10 if profile["id"] == "a" else 40
        return {"profile_id": profile["id"], "metrics": {"throughput": value, "latency_p95_seconds": 2},
                "throughput": value, "requests": [], "invalid_reasons": [], "valid": True}
    monkeypatch.setattr(performance, "measure", fake)
    workload = {"profiles": [{"id": "a", "weight": 1}, {"id": "b", "weight": 1}]}
    record = measure_suite(None, workload, tmp_path / "samples.jsonl", "v", 0, {})
    assert record["metrics"]["throughput"] == pytest.approx(20)
    assert record["metrics"]["latency_p95_seconds"] == 2
    assert set(record["profile_records"]) == {"a", "b"}


def test_aggregate_gain_cannot_hide_a_profile_regression():
    def row(short, long):
        return {"valid": True, "metrics": {"throughput": (short * long) ** .5}, "requests": [],
                "profile_weights": {"short": 1, "long": 1},
                "profile_records": {"short": {"throughput": short}, "long": {"throughput": long}}}
    result = decide({"passed": True}, [row(10, 10)] * 5, [row(20, 8)] * 5,
                    {"min_gain": .02, "max_profile_regression": .03})
    assert result.outcome == "reject" and result.reason == "profile_regression:long"


def test_gpu_annotation_duplicates_do_not_overwrite_cpu_phase_or_become_kernels():
    def event(name, device, cpu, gpu):
        return SimpleNamespace(key=name, device_type="DeviceType." + device, count=1,
            self_cpu_time_total=cpu, cpu_time_total=cpu, self_device_time_total=gpu, device_time_total=gpu)
    phases, cpu, kernels, operators = _torch_summaries([
        event("vibeforge.decode", "CPU", 900, 200), event("vibeforge.decode", "CUDA", 0, 900),
        event("aten::mm", "CPU", 50, 200), event("matmul_kernel", "CUDA", 0, 200)])
    assert len(phases) == 1 and phases[0]["cpu_total_seconds"] == .0009
    assert [r["name"] for r in kernels] == ["matmul_kernel"]
    assert [r["name"] for r in operators] == ["aten::mm"]


def test_concurrency_diagnostic_reports_actual_load_without_claiming_batching():
    class Client:
        def generate(self, **kwargs):
            return {"input_ids": [1], "token_ids": [2, 3],
                    "host_observation": {"queue_seconds": 0.01}}

    report = diagnose_concurrency(Client(), {
        "id": "concurrent_2", "prompts": ["a", "b"], "max_new_tokens": 2, "concurrency": 2})

    assert report["request_count"] == 4 and report["output_tokens"] == 8
    assert report["throughput"] > 0 and report["queue_p95_seconds"] == .01
    assert report["observed_batch_size_max"] is None and report["admission"] is False


def test_invalid_profile_throughput_preserves_evidence_instead_of_log_crash(monkeypatch, tmp_path):
    import vibeforge.evaluation.performance as performance
    monkeypatch.setattr(performance, "measure", lambda client, profile, *a, **kw: {
        "profile_id": profile["id"], "metrics": {"throughput": 0}, "throughput": 0,
        "requests": [], "invalid_reasons": ["invalid_throughput"], "valid": False})
    record = measure_suite(None, {"profiles": [{"id": "a", "weight": 1}]},
                           tmp_path / "samples.jsonl", "v", 0, {})
    assert not record["valid"] and record["throughput"] is None
    assert "invalid_profile_throughput" in record["invalid_reasons"]
