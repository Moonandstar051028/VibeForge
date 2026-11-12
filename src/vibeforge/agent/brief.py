"""Build a short, host-owned optimization brief from frozen evidence."""

from collections import Counter

from vibeforge.agent.candidate import (
    UNRESOLVED_PRECONDITION_REASONS, counterexample_reason,
)


def _operator_family(name):
    value = str(name or "").casefold()
    if "gemm_4bit" in value or "bitsandbytes" in value:
        return "bitsandbytes_nf4_gemm"
    if "cudalaunchkernel" in value:
        return "cuda_kernel_launch"
    if "copy" in value or "memcpy" in value:
        return "tensor_copy"
    if any(term in value for term in ("elementwise", "aten::mul", "aten::add")):
        return "elementwise"
    if any(term in value for term in ("gemm", "aten::mm", "aten::bmm", "matmul")):
        return "matrix_multiply"
    return str(name or "unknown")[:80]


def build_optimization_brief(context):
    observation = context.get("performance_observation") or {}
    runtime = observation.get("runtime") or context.get("effective_runtime") or {}
    features = context.get("model_features") or {}
    layers = features.get("layer_types") or []
    layer_counts = Counter(layers)
    measurements = observation.get("measurements") or {}
    phase_rows = []
    for name, measurement in measurements.items():
        value = measurement.get("value") if isinstance(measurement, dict) else None
        if (name.startswith("normal.phases.") and name.endswith("_seconds")
                and isinstance(value, (int, float)) and value > 0):
            phase_rows.append((value, name, measurement.get("unit")))
    phase_rows.sort(reverse=True)
    wall = (measurements.get("normal.backend_wall_seconds") or {}).get("value")
    bottleneck = None
    if phase_rows:
        value, metric, unit = phase_rows[0]
        bottleneck = {
            "metric": metric, "value": value, "unit": unit,
            "request_time_share_pct": round(value / wall * 100, 2)
            if isinstance(wall, (int, float)) and wall > 0 else None,
        }
    operators = []
    seen = set()
    for key in ("top_device_operator_ops", "top_device_ops", "top_cpu_ops"):
        for row in observation.get(key) or []:
            family = _operator_family(row.get("name"))
            if family in seen:
                continue
            seen.add(family)
            operators.append({
                "family": family, "calls": row.get("calls"),
                "self_device_seconds": row.get("self_device_seconds"),
                "self_cpu_seconds": row.get("self_cpu_seconds"),
            })
            if len(operators) == 5:
                break
        if len(operators) == 5:
            break
    # Keep independent rankings: filling a device-only list must not hide
    # CPU launch, synchronization, allocation, or other measured host costs.
    operator_evidence = {}
    for key, clock in (("top_cpu_ops", "self_cpu_seconds"),
                       ("top_device_operator_ops", "self_device_seconds"),
                       ("top_device_ops", "self_device_seconds")):
        rows = [row for row in observation.get(key) or []
                if isinstance(row.get(clock), (int, float)) and row[clock] > 0]
        rows.sort(key=lambda row: row[clock], reverse=True)
        operator_evidence[key] = [{
            "name": str(row.get("name", ""))[:200],
            "family": _operator_family(row.get("name")),
            "calls": row.get("calls"), "clock": clock, "seconds": row[clock],
        } for row in rows[:5]]
    operator_evidence["scope"] = (
        "Independent retained CPU, device-operator and device-kernel rankings. "
        "Rows and domains can overlap; do not add their times or treat profiler timings as admission evidence.")
    packages = {**context.get("environment", {}).get("inference", {}).get("packages", {}),
                **(runtime.get("packages") or {})}
    dependencies = {name: packages.get(name) for name in (
        "torch", "transformers", "bitsandbytes", "triton", "vllm",
        "flash-attn", "flash-linear-attention", "causal-conv1d")}
    constraints = []
    if runtime.get("continuous_batching") is False:
        constraints.append({
            "fact": "continuous_batching_disabled", "severity": "hard",
            "implication": "Removing the request lock is not batching. Check scheduler_capabilities for an existing static microbatch path; token-level continuous batching requires another implementation.",
        })
    compatibility = runtime.get("compatibility") or {}
    for name, record in compatibility.items():
        if isinstance(record, dict) and record.get("supported") is False:
            constraints.append({"fact": f"{name}_unsupported", "severity": "hard",
                                "implication": record.get("reason")})
    unavailable = list((observation.get("normal_request") or {}).get("unavailable_metrics") or [])[:12]
    attempted = list((context.get("candidate_contract") or {}).get("attempted_strategies") or [])
    strategy_history = [{key: row.get(key) for key in (
        "history_ref", "strategy_family", "mechanism_id", "change", "decision"
    )} for row in attempted[-12:]]
    do_not_repeat = [{
        "history_ref": row.get("history_ref"),
        "strategy_family": row.get("strategy_family"),
        "mechanism_id": row.get("mechanism_id"),
        "reason": (row.get("decision") or {}).get("reason"),
        "change": row.get("change"),
        "scope": (
            "This recorded implementation and configuration only. Different cache backends, batch sizes "
            "or implementations remain testable with a new evidence-bound prediction. "
            "No significant gain is inconclusive, not proof that the entire strategy is ineffective."
        ),
    } for row in attempted if counterexample_reason((row.get("decision") or {}).get("reason"))][-12:]
    unresolved_preconditions = [{
        "history_ref": row.get("history_ref"),
        "mechanism_id": row.get("mechanism_id"),
        "reason": (row.get("decision") or {}).get("reason"),
        "required_evidence": "diagnostic.concurrent.*.throughput from a workload profile with concurrency >= 2",
        "instruction": "Do not submit this mechanism again until the required evidence is present.",
    } for row in attempted
        if (row.get("decision") or {}).get("reason") in UNRESOLVED_PRECONDITION_REASONS][-12:]
    allowed = set((context.get("modification_surfaces") or {}).get("allowed") or [])
    directions = []
    if "scheduler" in allowed:
        has_concurrent_workload = any(
            isinstance(profile.get("concurrency"), int) and profile["concurrency"] >= 2
            for profile in ((context.get("workload") or {}).get("profiles")
                            or [context.get("workload") or {}])
            if isinstance(profile, dict)
        )
        scheduler_blocked = not has_concurrent_workload or not any(
            name.startswith("diagnostic.concurrent.") and name.endswith(".throughput")
            for name in measurements)
        directions.append({
            "surface": "scheduler",
            "status": ("blocked_missing_concurrency_evidence" if scheduler_blocked else
                       "eligible"),
            "required_evidence": "concurrent throughput, queue time, latency and observed batch size",
        })
    if "cache_policy" in allowed:
        directions.append({
            "surface": "cache_policy",
            "status": "eligible_with_correctness_risk",
            "blocked_mechanisms": [],
            "required_evidence": "observed cache class, decode metric and request-isolation correctness",
        })
    if "execution_adapter" in allowed:
        directions.append({"surface": "execution_adapter", "status": "eligible",
                           "required_evidence": "runtime identity and end-to-end paired measurement"})
    if "native_kernel" in allowed:
        directions.append({"surface": "native_kernel", "status": "eligible_with_build_recipe",
                           "required_evidence": "operator family, shapes/call count and component microbenchmark"})
    failures = []
    for row in (context.get("history") or [])[-6:]:
        decision = row.get("decision") or {}
        if decision.get("outcome") != "accept":
            failures.append({"history_ref": row.get("history_ref"), "reason": decision.get("reason"),
                             "change": (row.get("change") or row.get("hypothesis") or "")[:180]})
    workload = context.get("workload") or {}
    profiles = workload.get("profiles") or [workload]
    profile_summary = [{key: profile.get(key) for key in (
        "id", "max_new_tokens", "concurrency", "requests_per_prompt", "weight")}
        for profile in profiles]
    evidence_paths = ((context.get("candidate_contract") or {}).get("observation_paths") or [])[:12]
    concurrency_evidence = [{
        "profile_id": profile_id, "concurrency": row.get("concurrency"),
        "throughput": row.get("throughput"), "latency_p95_seconds": row.get("latency_p95_seconds"),
        "queue_p95_seconds": row.get("queue_p95_seconds"),
        "observed_batch_size_max": row.get("observed_batch_size_max"),
    } for profile_id, row in list((observation.get("concurrency_diagnostics") or {}).items())[:12]]
    return {
        "schema_version": 1,
        "decision_rule": "Choose only a mechanism whose required evidence is available; otherwise report blocked.",
        "current_state": {
            "stable_ref": context.get("stable_ref"),
            "source_of_truth": "The stable workspace and observed runtime below; rejected candidates are not the baseline.",
            "stable_engine_config": context.get("stable_engine_config") or {},
            "observed_runtime": {
                "execution_path": runtime.get("execution_path"),
                "request_scheduler": runtime.get("request_scheduler"),
                "batching_mode": runtime.get("batching_mode"),
                "attention_implementations": runtime.get("attention_implementations"),
                "observed_cache_class": (observation.get("normal_request") or {}).get("observed_cache_class"),
                "observed_generate_kwargs": (observation.get("normal_request") or {}).get("observed_generate_kwargs"),
            },
            "previous_attempt": ({
                "history_ref": f"{context.get('campaign_id')}:{(context.get('previous') or {}).get('iteration')}",
                "candidate": (context.get("previous") or {}).get("candidate"),
                "decision": (context.get("previous") or {}).get("decision"),
                "promoted_to_stable": (context.get("previous") or {}).get("candidate") == context.get("stable_ref"),
            } if (context.get("previous") or {}).get("iteration") is not None else None),
        },
        "objective": {
            "metric": (context.get("requirements") or {}).get("objective"),
            "min_gain_pct": (context.get("protocol") or {}).get("min_gain", 0) * 100,
            "max_profile_regression_pct": (context.get("protocol") or {}).get("max_profile_regression", 0) * 100,
        },
        "model_summary": {
            "model_type": features.get("model_type"), "hidden_size": features.get("hidden_size"),
            "intermediate_size": features.get("intermediate_size"),
            "num_hidden_layers": features.get("num_hidden_layers"),
            "layer_type_counts": dict(layer_counts), "head_dim": features.get("head_dim"),
            "max_position_embeddings": features.get("max_position_embeddings"),
        },
        "runtime_facts": {
            "execution_path": runtime.get("execution_path"),
            "request_scheduler": runtime.get("request_scheduler"),
            "continuous_batching": runtime.get("continuous_batching"),
            "batching_mode": runtime.get("batching_mode"),
            "scheduler_capabilities": runtime.get("scheduler_capabilities"),
            "attention_implementations": runtime.get("attention_implementations"),
            "precision": runtime.get("runtime_precision") or context.get("precision"),
            "device": runtime.get("model_device") or (context.get("model") or {}).get("device"),
            "dependencies": dependencies,
        },
        "workload_profiles": profile_summary,
        "primary_bottleneck": bottleneck,
        "top_operator_families": operators,
        "operator_evidence": operator_evidence,
        "implementation_index": observation.get("implementation_index") or {},
        "profiler_diagnostics": {
            key: value for key, value in list((observation.get("diagnostics") or {}).items())[:12]
        },
        "concurrency_evidence": concurrency_evidence,
        "hard_constraints": constraints,
        "unavailable_metrics": unavailable,
        "modification_directions": directions,
        "strategy_history": strategy_history,
        "do_not_repeat": do_not_repeat,
        "unresolved_preconditions": unresolved_preconditions,
        "relevant_failures": failures,
        "evidence_paths": evidence_paths,
        "details": "Full model, hardware, environment, profile and history remain in context.json and bound evidence.",
    }
