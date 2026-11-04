"""Identical-path performance, client timing, concurrency and external resource metrics."""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from vibeforge.evaluation.correctness import compare_generation
from vibeforge.evaluation.statistics import percentile
from vibeforge.observability.logs import append_record
from vibeforge.profiling.resources import ResourceSampler, descendants
from vibeforge.profiling.sampler import sample
from vibeforge.storage.atomic import digest


def measure(client, workload, output, version, group, protocol, allowed_pids=(),
            reference=None, startup_seconds=None, source_bytes=None, attempt=0):
    before, reasons = sample(), []
    gpu_required = protocol.get("require_exclusive_gpu") and protocol.get("device_backend", "cuda") != "cpu"
    allowed = descendants(allowed_pids)

    def check_environment(snapshot):
        if gpu_required:
            processes = snapshot["gpu_processes"]
            if processes is None:
                reasons.append("gpu_process_inventory_unavailable")
            elif any(p["pid"] not in allowed for p in processes):
                reasons.append("external_gpu_process")
        limit = protocol.get("max_temperature_c")
        if limit is not None:
            temperature = snapshot.get("temperature_c")
            if temperature is None:
                reasons.append("temperature_inventory_unavailable")
            elif temperature > limit:
                reasons.append("temperature_limit")

    check_environment(before)
    expected = (reference or {}).get("benchmark_cases", [])
    if not expected:
        reasons.append("benchmark_reference_missing")

    def request(item):
        index, prompt = item
        started = time.perf_counter()
        result = client.generate(prompt=prompt, max_new_tokens=workload["max_new_tokens"], fixed_length=True)
        elapsed = time.perf_counter() - started
        case = next((c for c in expected if c["prompt"] == prompt and c["max_new_tokens"] == workload["max_new_tokens"]), None)
        valid = bool(case) and compare_generation(result, case, protocol)
        return {"index": index, "latency_seconds": elapsed, "tokens": len(result["token_ids"]),
                "output_hash": digest(result["token_ids"]), "correct": valid,
                "backend_metrics": {key: result.get(key) for key in ("elapsed_seconds", "ttft_seconds", "tpot_seconds")}}

    jobs = list(enumerate(workload["prompts"] * workload.get("concurrency", 1)))
    with ResourceSampler(allowed_pids, protocol.get("resource_sample_interval", 0.05)) as resources:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workload.get("concurrency", 1)) as pool:
            requests = list(pool.map(request, jobs))
        elapsed = time.perf_counter() - started
    if any(r["tokens"] != workload["max_new_tokens"] for r in requests):
        reasons.append("output_work_mismatch")
    if any(not r["correct"] for r in requests):
        reasons.append("benchmark_output_incorrect")
    after = sample()
    check_environment(after)
    tokens = sum(r["tokens"] for r in requests)
    throughput = tokens / elapsed
    if not math.isfinite(throughput) or throughput <= 0:
        reasons.append("invalid_throughput")
    metrics = {"throughput": throughput,
               "latency_p95_seconds": percentile([r["latency_seconds"] for r in requests], 0.95),
               "peak_rss_bytes": resources.peak, "startup_seconds": startup_seconds, "source_bytes": source_bytes}
    for name in {protocol.get("objective", "throughput"), *protocol.get("limits", {})}:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            reasons.append(f"required_metric_unavailable:{name}")
    record = {"version": version, "group": group, "attempt": attempt,
              "tokens": tokens, "elapsed_seconds": elapsed, "throughput": throughput,
              "metrics": metrics, "requests": requests,
              "concurrency": workload.get("concurrency", 1), "device_before": before, "device_after": after,
              "resource_sampling": {"interval_seconds": resources.interval, "samples": resources.samples,
                                    "scope": "engine process tree RSS; shared pages may be counted twice; excludes device-only VRAM"},
              "valid": not reasons, "invalid_reasons": sorted(set(reasons))}
    append_record(output, record)
    return record
