"""Measured request performance with raw samples and environment evidence."""

import math
import time
from vibeforge.observability.logs import append_record
from vibeforge.profiling.sampler import sample


def measure(client, workload, output, version, group, protocol, allowed_pids=()):
    before = sample()
    reasons = []
    if protocol.get("require_exclusive_gpu"):
        processes = before["gpu_processes"]
        if processes is None:
            reasons.append("gpu_process_inventory_unavailable")
        elif any(p["pid"] not in allowed_pids for p in processes):
            reasons.append("external_gpu_process")
    limit = protocol.get("max_temperature_c")
    if limit is not None and before["temperature_c"] is not None and before["temperature_c"] > limit:
        reasons.append("temperature_limit")
    started = time.perf_counter()
    requests = []
    for prompt in workload["prompts"]:
        result = client.generate(prompt=prompt, max_new_tokens=workload["max_new_tokens"], fixed_length=True)
        count = len(result["token_ids"])
        if count != workload["max_new_tokens"]:
            reasons.append("output_work_mismatch")
        requests.append({key: result.get(key) for key in ("elapsed_seconds", "ttft_seconds", "tpot_seconds")})
    elapsed = time.perf_counter() - started
    after = sample()
    if protocol.get("require_exclusive_gpu"):
        processes = after["gpu_processes"]
        if processes is None or any(p["pid"] not in allowed_pids for p in processes):
            reasons.append("external_gpu_process_or_unknown_inventory")
    if limit is not None and after["temperature_c"] is not None and after["temperature_c"] > limit:
        reasons.append("temperature_limit")
    tokens = len(requests) * workload["max_new_tokens"]
    throughput = tokens / elapsed
    if not math.isfinite(throughput) or throughput <= 0:
        reasons.append("invalid_throughput")
    record = {"version": version, "group": group, "tokens": tokens, "elapsed_seconds": elapsed,
              "throughput": throughput, "requests": requests, "device_before": before,
              "device_after": after, "valid": not reasons, "invalid_reasons": sorted(set(reasons))}
    append_record(output, record)
    return record
