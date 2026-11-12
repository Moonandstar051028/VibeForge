"""Identical-path performance, client timing, concurrency and external resource metrics."""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from vibeforge.evaluation.correctness import compare_generation
from vibeforge.evaluation.statistics import percentile
from vibeforge.observability.logs import append_record
from vibeforge.profiling.resources import DeviceTelemetrySampler, ResourceSampler, descendants
from vibeforge.profiling.sampler import sample
from vibeforge.storage.atomic import digest
from vibeforge.contracts.workload import workload_profiles


def measurement_mode(protocol):
    """Use identical lightweight timing for throughput and end-to-end latency."""
    required = {protocol.get("objective", "throughput"), *protocol.get("limits", {}),
                *protocol.get("absolute_limits", {})}
    detailed = ("ttft", "tpot", "prefill_forward", "decode_forward", "cuda_peak")
    return "detailed" if any(name.startswith(detailed) for name in required) else "wall"


def diagnose_concurrency(client, workload):
    """Run a small frozen concurrent load for mechanism feedback, never admission."""
    concurrency = int(workload.get("concurrency", 1))
    if concurrency < 2:
        raise ValueError("Concurrency diagnostics require concurrency >= 2")
    jobs = workload["prompts"] * concurrency

    def request(prompt):
        started = time.perf_counter()
        result = client.generate(
            prompt=prompt, max_new_tokens=workload["max_new_tokens"], fixed_length=True)
        return result, time.perf_counter() - started

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(request, jobs))
    elapsed = time.perf_counter() - started
    tokens = sum(len(result["token_ids"]) for result, _ in rows)
    latencies = [latency for _, latency in rows]
    observations = [result.get("host_observation") or {} for result, _ in rows]
    queues = [row.get("queue_seconds") for row in observations]
    batch_sizes = [result.get("batch_size") for result, _ in rows]
    batch_sizes = [value for value in batch_sizes if isinstance(value, int) and value > 0]
    return {
        "schema_version": 1, "kind": "concurrency_diagnostic", "admission": False,
        "profile_id": workload.get("id", "default"), "concurrency": concurrency,
        "request_count": len(rows), "output_tokens": tokens, "elapsed_seconds": elapsed,
        "throughput": tokens / elapsed if elapsed > 0 else None,
        "latency_p50_seconds": percentile(latencies, .5),
        "latency_p95_seconds": percentile(latencies, .95),
        "queue_p95_seconds": (percentile(queues, .95)
                              if queues and all(isinstance(value, (int, float)) for value in queues) else None),
        "observed_batch_size_max": max(batch_sizes) if batch_sizes else None,
        "limitations": [
            "Small diagnostic load; excluded from formal admission statistics.",
            "A missing observed batch size means the backend did not expose batching, not that batch size was one.",
        ],
    }


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
            temperature = snapshot.get("gpu_temperature_c" if protocol.get("device_backend", "cuda") != "cpu" else "temperature_c")
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
        result = client.generate(prompt=prompt, max_new_tokens=workload["max_new_tokens"], fixed_length=True,
                                 observation_mode=protocol.get("measurement_mode", measurement_mode(protocol)))
        elapsed = time.perf_counter() - started
        return {"index": index, "latency_seconds": elapsed, "result": result, "prompt": prompt}

    def validate(row):
        result, prompt = row["result"], row["prompt"]
        case = next((c for c in expected if c["prompt"] == prompt and c["max_new_tokens"] == workload["max_new_tokens"]), None)
        valid = bool(case) and compare_generation(result, case, protocol)
        length = len(result["input_ids"])
        valid = valid and workload.get("min_input_tokens", 1) <= length <= workload.get("max_input_tokens", math.inf)
        return {"index": row["index"], "latency_seconds": row["latency_seconds"], "tokens": len(result["token_ids"]),
                "input_tokens": length, "output_hash": digest(result["token_ids"]), "correct": valid,
                "host_observation": result.get("host_observation"),
                "backend_metrics": {key: result.get(key) for key in ("elapsed_seconds", "ttft_seconds", "tpot_seconds")}}

    jobs = list(enumerate(workload["prompts"] * workload.get("concurrency", 1) * workload.get("requests_per_prompt", 1)))
    device_enabled = protocol.get("device_backend", "cuda") == "cuda"
    device_resources = DeviceTelemetrySampler(
        protocol.get("device_sample_interval", 1.0), enabled=device_enabled)
    with (ResourceSampler(allowed_pids, protocol.get("resource_sample_interval", 0.05)) as resources,
          device_resources):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workload.get("concurrency", 1)) as pool:
            requests = list(pool.map(request, jobs))
        elapsed = time.perf_counter() - started
    device_sampling = device_resources.report()
    # Reference comparison and hashing are deliberately outside the timed window.
    requests = [validate(row) for row in requests]
    if any(r["tokens"] != workload["max_new_tokens"] for r in requests):
        reasons.append("output_work_mismatch")
    if any(not r["correct"] for r in requests):
        reasons.append("benchmark_output_incorrect")
    after = sample()
    check_environment(after)
    temperature_limit = protocol.get("max_temperature_c")
    temperature_peaks = [
        device["metrics"]["temperature_c"]["maximum"]
        for device in device_sampling["devices"]
        if "temperature_c" in device["metrics"]
    ]
    if temperature_limit is not None and temperature_peaks and max(temperature_peaks) > temperature_limit:
        reasons.append("temperature_limit")
    if protocol.get("require_continuous_device_telemetry") and device_sampling["status"] != "collected":
        reasons.append("continuous_device_telemetry_unavailable")
    tokens = sum(r["tokens"] for r in requests)
    throughput = tokens / elapsed
    if not math.isfinite(throughput) or throughput <= 0:
        reasons.append("invalid_throughput")
    metrics = {"throughput": throughput,
               "latency_p95_seconds": percentile([r["latency_seconds"] for r in requests], 0.95),
               "peak_rss_bytes": resources.peak, "startup_seconds": startup_seconds, "source_bytes": source_bytes}
    for name in ("ttft_seconds", "tpot_seconds", "queue_seconds"):
        values = [(row.get("host_observation") or {}).get(name) for row in requests]
        metrics[name + "_p95"] = percentile(values, .95) if values and all(isinstance(v, (int, float)) for v in values) else None
    for name in ("prefill_forward_seconds", "decode_forward_seconds"):
        values = ((row.get("host_observation") or {}).get("phases", {}).get(name) for row in requests)
        values = list(values)
        metrics[name + "_p95"] = percentile(values, .95) if values and all(isinstance(v, (int, float)) for v in values) else None
    for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
        values = [((row.get("host_observation") or {}).get("cuda_memory") or {}).get(name) for row in requests]
        metrics["cuda_" + name] = max(values) if values and all(isinstance(v, (int, float)) for v in values) else None
    metrics.update(latency_p50_seconds=percentile([r["latency_seconds"] for r in requests], .5),
                   latency_p99_seconds=percentile([r["latency_seconds"] for r in requests], .99))
    for name in {protocol.get("objective", "throughput"), *protocol.get("limits", {})}:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            reasons.append(f"required_metric_unavailable:{name}")
    min_tail_requests = protocol.get("min_tail_requests", 100)
    tail_latency = {
        "request_count": len(requests), "minimum_reliable_requests": min_tail_requests,
        "sample_size_sufficient": len(requests) >= min_tail_requests,
        "scope": "request quantiles within this profile and measurement group",
    }
    record = {"version": version, "group": group, "attempt": attempt,
              "measurement_mode": protocol.get("measurement_mode", measurement_mode(protocol)),
              "tokens": tokens, "elapsed_seconds": elapsed, "throughput": throughput,
              "metrics": metrics, "requests": requests,
              "concurrency": workload.get("concurrency", 1), "device_before": before, "device_after": after,
              "device_sampling": device_sampling,
              "resource_sampling": {"interval_seconds": resources.interval, "samples": resources.samples,
                                    "scope": "engine process tree RSS; shared pages may be counted twice; excludes device-only VRAM"},
              "valid": not reasons, "invalid_reasons": sorted(set(reasons))}
    record.update(profile_id=workload.get("id", "default"), request_count=len(requests),
                  tail_latency=tail_latency,
                  tail_latency_warning=(
                      "Descriptive quantiles only; too few requests for reliable tail estimates"
                      if not tail_latency["sample_size_sufficient"] else None))
    append_record(output, record)
    return record


def measure_suite(client, workload, output, version, group, protocol, allowed_pids=(), **kwargs):
    if not workload.get("profiles"):
        return measure(client, workload, output, version, group, protocol, allowed_pids, **kwargs)
    profiles = workload_profiles(workload)
    # Overall constraints are applied to the aggregate; profiles still require
    # valid output, environment and positive throughput independently.
    profile_protocol = {**protocol, "objective": "throughput", "limits": {},
                        "measurement_mode": measurement_mode(protocol)}
    records = [measure(client, profile, output, version, group, profile_protocol, allowed_pids, **kwargs)
               for profile in profiles]
    weight = sum(profile["weight"] for profile in profiles)
    measurable = all(isinstance(r["throughput"], (int, float)) and math.isfinite(r["throughput"])
                     and r["throughput"] > 0 for r in records)
    score = math.exp(sum(profile["weight"] * math.log(record["throughput"])
                         for profile, record in zip(profiles, records)) / weight) if measurable else None
    names = set().union(*(record["metrics"] for record in records))
    metrics = {name: max(record["metrics"][name] for record in records)
               if all(isinstance(record["metrics"].get(name), (int, float)) for record in records) else None
               for name in names}
    metrics["throughput"] = score
    reasons = [reason for record in records for reason in record["invalid_reasons"]]
    if not measurable:
        reasons.append("invalid_profile_throughput")
    for name in {protocol.get("objective", "throughput"), *protocol.get("limits", {})}:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            reasons.append("required_metric_unavailable:" + name)
    record = {"version": version, "group": group, "attempt": kwargs.get("attempt", 0),
              "metrics": metrics, "throughput": score, "profile_records": {r["profile_id"]: r for r in records},
              "profile_weights": {p["id"]: p["weight"] for p in profiles},
              "aggregation": "weighted geometric mean of profile output-token throughput; not pooled tokens/second; other metrics use worst profile",
              "requests": [request for row in records for request in row["requests"]],
              "valid": not reasons, "invalid_reasons": sorted(set(reasons))}
    append_record(output, record)
    return record
