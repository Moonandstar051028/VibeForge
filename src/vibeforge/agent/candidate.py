"""Shared proposal contract, rechecked by the controller before any experiment."""

import ast
import json
import math
from pathlib import Path

from vibeforge.storage.atomic import digest, read_json
from vibeforge.storage.paths import confined
from vibeforge.versioning.artifacts import inventory, source_hash
from vibeforge.agent.backend_policy import attention_backend_issues, cache_implementation_issues
from vibeforge.contracts.requirements import OBJECTIVES


COUNTEREXAMPLE_REASONS = {
    "correctness_failed", "quality_failed", "ineffective_runtime_change",
    "duplicate_runtime_effect", "performance_regression", "no_significant_gain",
    "profile_regression", "candidate_probe_failed",
}

UNRESOLVED_PRECONDITION_REASONS = {
    "missing_scheduler_concurrency_evidence",
}

RESERVED_CANDIDATE_ROOTS = {
    ".vibeforge", "campaigns", "final-analysis", "iterations",
    "knowledge-drafts", "logs", "reports", "var",
}


def strategy_from_plan(plan):
    """Classify an historical plan into a stable strategy family and mechanism."""
    plan = plan or {}
    text = " ".join(str(plan.get(key) or "") for key in (
        "measured_bottleneck", "cause", "change", "mechanism_surface",
    )).casefold()
    if any(term in text for term in ("cache_implementation", "kv cache", "cache policy")):
        return {"strategy_family": "cache_policy", "mechanism_id": "cache_implementation_switch"}
    if "use_cache" in text or "disable cache" in text:
        return {"strategy_family": "cache_policy", "mechanism_id": "cache_enablement"}
    if any(term in text for term in ("attn_implementation", "attention backend", "flash attention")):
        return {"strategy_family": "attention_backend", "mechanism_id": "attention_backend_switch"}
    if any(term in text for term in ("request_scheduler", "microbatch", "batching", "scheduler")):
        return {"strategy_family": "scheduler", "mechanism_id": "request_scheduler_change"}
    if any(term in text for term in ("native kernel", "cuda kernel", "triton", "gemm kernel")):
        return {"strategy_family": "native_kernel", "mechanism_id": "native_kernel_change"}
    return {"strategy_family": "execution_adapter", "mechanism_id": "execution_adapter_change"}


def candidate_strategy(changed_files, changed_fields):
    """Classify the actual candidate diff rather than trusting its prose label."""
    files = set(changed_files or [])
    fields = set(changed_fields or [])
    source_files = files - {"engine.json"}
    if any(Path(name).suffix in {".cu", ".cpp", ".c", ".h", ".hpp"} for name in source_files):
        return {"strategy_family": "native_kernel", "mechanism_id": "native_kernel_change"}
    # Configuration fields identify these mechanisms more precisely than
    # accompanying implementation or metadata files do.
    if fields and fields <= {"cache_implementation"}:
        return {"strategy_family": "cache_policy", "mechanism_id": "cache_implementation_switch"}
    if fields and fields <= {"use_cache"}:
        return {"strategy_family": "cache_policy", "mechanism_id": "cache_enablement"}
    if fields and fields <= {"attn_implementation"}:
        return {"strategy_family": "attention_backend", "mechanism_id": "attention_backend_switch"}
    if fields and fields <= {"request_scheduler", "microbatch_max_size", "microbatch_wait_ms"}:
        return {"strategy_family": "scheduler", "mechanism_id": "request_scheduler_change"}
    return {"strategy_family": "execution_adapter", "mechanism_id": "execution_adapter_change"}


def counterexample_reason(reason):
    return reason in COUNTEREXAMPLE_REASONS or str(reason or "").startswith("profile_regression:")


def observation_benefit_bounds(observation):
    """Compute conservative Amdahl bounds from ordinary, non-profiled phase clocks."""
    measurements = (observation or {}).get("measurements") or {}
    wall = measurements.get("normal.backend_wall_seconds") or {}
    wall_value = wall.get("value")
    if not isinstance(wall_value, (int, float)) or wall_value <= 0:
        return []
    bounds = []
    for metric, measurement in measurements.items():
        value = measurement.get("value") if isinstance(measurement, dict) else None
        if (not metric.startswith("normal.phases.") or not metric.endswith("_seconds")
                or not isinstance(value, (int, float)) or value <= 0):
            continue
        raw_share = value / wall_value
        share = min(1.0, raw_share)
        bounds.append({
            "metric": metric, "value_seconds": value,
            "request_wall_seconds": wall_value, "request_time_share": share,
            "maximum_throughput_improvement_pct": (
                share / (1 - share) * 100 if share < 1 else None),
            "maximum_latency_reduction_pct": share * 100,
            "bound_scope": "ordinary request phase divided by ordinary backend wall time",
            "warning": ("Phase interval meets or exceeds wall time; overlap or clock scope prevents a finite bound."
                        if raw_share >= 1 else None),
        })
    return sorted(bounds, key=lambda row: row["request_time_share"], reverse=True)


def _benefit_bound(metric, measurement, measurements, direction, expected_pct):
    wall = measurements.get("normal.backend_wall_seconds") or {}
    wall_value = wall.get("value")
    value = measurement.get("value") if isinstance(measurement, dict) else None
    if metric == "normal.backend_wall_seconds":
        return {"available": True, "bottleneck_time_share": 1.0,
                "maximum_overall_improvement_pct": None, "feasible": True,
                "reason": "The reference covers the complete request wall time."}
    if (not isinstance(metric, str) or not metric.startswith("normal.phases.")
            or not metric.endswith("_seconds") or not isinstance(value, (int, float))
            or not isinstance(wall_value, (int, float)) or value <= 0 or wall_value <= 0):
        return {"available": False, "feasible": None,
                "reason": "The selected metric is not an additive ordinary-request phase duration."}
    raw_share = value / wall_value
    if raw_share >= 1:
        return {"available": False, "feasible": None, "bottleneck_time_share": 1.0,
                "reason": "The phase meets or exceeds request wall time; overlap prevents a finite bound."}
    share = max(0.0, raw_share)
    maximum = (share / (1 - share) * 100 if direction == "increase" else share * 100)
    fraction = expected_pct / 100
    required = ((1 - 1 / (1 + fraction)) / share * 100
                if direction == "increase" else fraction / share * 100)
    return {
        "available": True, "bottleneck_time_share": share,
        "maximum_overall_improvement_pct": maximum,
        "required_bottleneck_elimination_pct": required,
        "feasible": expected_pct <= maximum + 1e-9,
        "reason": "Amdahl upper bound from the ordinary-request bottleneck time share.",
    }


def history_reference(campaign_id, iteration):
    """Return an unambiguous reference suitable for model/tool contracts."""
    return f"{campaign_id}:{int(iteration)}"


def semantic_fingerprint(directory):
    values = {}
    files = inventory(directory)
    for name in files:
        path = Path(directory) / name
        if path.suffix not in {".py", ".json", ".cu", ".cpp", ".c", ".h", ".hpp"}:
            continue
        if path.suffix == ".py":
            values[name] = ast.dump(ast.parse(path.read_text()), include_attributes=False)
        elif path.suffix == ".json":
            values[name] = json.loads(path.read_text())
        else:
            values[name] = files[name]
    return digest(values)


def runtime_effect_fingerprint(observation):
    """Fingerprint observable execution semantics, excluding timing noise."""
    observation = observation or {}
    runtime = observation.get("runtime") or {}
    normal = observation.get("normal_request") or {}
    kwargs = [{key: value for key, value in row.items()
               if key not in {"max_new_tokens", "min_new_tokens"}}
              for row in (normal.get("observed_generate_kwargs") or [])]
    effect = {
        "execution_path": runtime.get("execution_path"),
        "attention_implementations": runtime.get("attention_implementations"),
        "precision": runtime.get("precision") or observation.get("precision"),
        "request_scheduler": runtime.get("request_scheduler"),
        "observed_cache_class": normal.get("observed_cache_class"),
        "observed_generate_kwargs": kwargs,
    }
    # A generic adapter path alone cannot distinguish mechanisms. Avoid false
    # duplicates when the engine exposes no cache/backend/generation details.
    if not (effect["attention_implementations"] or
            effect["observed_cache_class"] or effect["observed_generate_kwargs"]):
        return None
    return digest(effect)


def _runtime_preconditions(runtime, prefix=""):
    """Expose the observed leaf fields in the same form accepted by the gate."""
    values = []
    for key, value in runtime.items():
        field = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            values.extend(_runtime_preconditions(value, field))
        else:
            values.append({"field": field, "equals": value})
    return values


def complete_experiment_plan(plan, context):
    """Fill controller-owned facts so the model only supplies scientific judgment."""
    if not isinstance(plan, dict):
        return plan
    value = dict(plan)
    contract = context.get("candidate_contract") or {}
    paths = (value["evidence_paths"] if "evidence_paths" in value
             else contract.get("observation_paths", [])[:1])
    value["evidence_paths"] = paths
    objective = context.get("requirements", {}).get("objective", "throughput")
    if contract.get("version", 1) >= 3:
        # The campaign objective is frozen host configuration, not model
        # judgment. A model may reasonably confuse a measurement key such as
        # normal.backend_wall_seconds with this field; never let that alter or
        # invalidate an otherwise bound v3 experiment.
        value["metric"] = objective
        metric_direction = OBJECTIVES.get(objective)
        value["direction"] = "increase" if metric_direction == "maximize" else "decrease"
    else:
        value.setdefault("metric", objective)
        metric_direction = OBJECTIVES.get(value["metric"]) if isinstance(value["metric"], str) else None
        value.setdefault("direction", "increase" if metric_direction == "maximize" else "decrease")
    records = {}
    for name in paths:
        try:
            path = confined(Path(context["evidence_root"]), name)
            if path.is_file():
                records[name] = read_json(path)
        except (KeyError, TypeError, ValueError, OSError):
            continue
    observation_paths = set(contract.get("observation_paths", []))
    evidence_path = next((name for name in paths if name in observation_paths), None)
    if evidence_path is None:
        evidence_path = next((name for name in contract.get("observation_paths", []) if name in records), None)
    record = records.get(evidence_path, {})
    measurements = record.get("measurements", {})
    requested = value.pop("bottleneck_metric", None)
    if requested is not None and requested not in measurements:
        raise ValueError(
            f"Unknown bottleneck_metric {requested!r} in {evidence_path!r}. "
            f"Select an observed metric: {sorted(measurements)}")
    if requested not in measurements:
        preferred = (
            "normal.phases.decode_forward_seconds",
            "normal.backend_wall_seconds",
            "normal.tpot_seconds",
            "normal.ttft_seconds",
        )
        requested = next((name for name in preferred if name in measurements), None)
    measurement = measurements.get(requested) if requested else None
    if measurement and (contract.get("version", 1) >= 3 or "bottleneck_reference" not in value):
        value["bottleneck_reference"] = {
            "evidence_path": evidence_path, "metric": requested,
            "value": measurement.get("value"), "unit": measurement.get("unit"),
        }
    profiles = context.get("workload", {}).get("profiles", [{"id": "default"}])
    if contract.get("version", 1) >= 3:
        value["target_profile_ids"] = [profile["id"] for profile in profiles]
    else:
        value.setdefault("target_profile_ids", [profile["id"] for profile in profiles])
    runtime = record.get("runtime", {})
    if runtime.get("execution_path") is not None and (
            contract.get("version", 1) >= 3 or "runtime_preconditions" not in value):
        value["runtime_preconditions"] = [{"field": "execution_path", "equals": runtime["execution_path"]}]
    expected = value.get("expected_change_pct")
    unit = (measurement or {}).get("unit")
    if contract.get("version", 1) >= 3:
        value["expected_bottleneck_change_pct"] = expected
        value["bottleneck_direction"] = "decrease" if unit in {"seconds", "bytes"} else "increase"
        value["expected_change_basis"] = value.get("cause", "")
    else:
        value.setdefault("expected_bottleneck_change_pct", expected)
        value.setdefault("bottleneck_direction", "decrease" if unit in {"seconds", "bytes"} else "increase")
        value.setdefault("expected_change_basis", value.get("cause", ""))
    if measurement and isinstance(expected, (int, float)):
        value["benefit_bound"] = _benefit_bound(
            requested, measurement, measurements, value.get("direction"), expected)
    return value


def candidate_gate(workspace, context, plan):
    contract = context.get("candidate_contract")
    if not contract:
        return {"valid": True, "legacy": True}
    parent = Path(contract["parent_source"])
    try:
        fingerprint = semantic_fingerprint(workspace)
        base_fingerprint = semantic_fingerprint(parent)
    except (ValueError, SyntaxError, OSError) as error:
        return {"valid": False, "reason": "invalid_source", "error": str(error)}
    result = {"valid": False, "fingerprint": fingerprint}
    if fingerprint == base_fingerprint:
        return {**result, "reason": "no_change"}
    before, after = inventory(parent), inventory(workspace)
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    reserved = [name for name in changed if Path(name).parts[0] in RESERVED_CANDIDATE_ROOTS]
    if reserved:
        return {**result, "reason": "reserved_candidate_artifact",
                "changed_files": reserved}
    allowed_surfaces = set(context.get("optimization", {}).get("allowed_surfaces", []))
    scheduler_fields = {"request_scheduler", "microbatch_max_size", "microbatch_wait_ms"}
    engine_config_changed = "engine.json" in changed
    changed_fields = set()
    if engine_config_changed:
        try:
            parent_config = json.loads((parent / "engine.json").read_text())
            candidate_config = json.loads((Path(workspace) / "engine.json").read_text())
        except (OSError, ValueError) as error:
            return {**result, "reason": "invalid_engine_config", "error": str(error)}
        changed_fields = {key for key in parent_config.keys() | candidate_config.keys()
                          if parent_config.get(key) != candidate_config.get(key)}
        if changed_fields & scheduler_fields:
            observation = context.get("performance_observation") or {}
            measurements = observation.get("measurements") or {}
            workload = context.get("workload") or {}
            workload_profiles = workload.get("profiles") or [workload]
            has_concurrent_profile = any(
                isinstance(profile.get("concurrency"), int) and profile["concurrency"] >= 2
                for profile in workload_profiles if isinstance(profile, dict)
            )
            has_concurrency_evidence = has_concurrent_profile and any(
                name.startswith("diagnostic.concurrent.") and name.endswith(".throughput")
                for name in measurements
            )
            for previous in contract.get("attempted_strategies", []):
                decision = previous.get("decision") or {}
                if (previous.get("mechanism_id") == "request_scheduler_change"
                        and decision.get("reason") in UNRESOLVED_PRECONDITION_REASONS
                        and not has_concurrency_evidence):
                    return {
                        **result,
                        "strategy_family": "scheduler",
                        "mechanism_id": "request_scheduler_change",
                        "reason": "unresolved_candidate_precondition",
                        "previous_history_ref": previous.get("history_ref"),
                        "previous_reason": decision.get("reason"),
                        "required_evidence": "diagnostic.concurrent.*.throughput for a workload with concurrency >= 2",
                    }
        required_surfaces = set()
        if changed_fields & {"cache_implementation", "use_cache", "attn_implementation"}:
            required_surfaces.add("cache_policy")
        if changed_fields & scheduler_fields:
            required_surfaces.add("scheduler")
        if changed_fields - {"cache_implementation", "use_cache", "attn_implementation"} - scheduler_fields:
            required_surfaces.add("execution_adapter")
        unauthorized = required_surfaces - allowed_surfaces
        if unauthorized:
            return {**result, "reason": "unauthorized_modification_surface",
                    "surfaces": sorted(unauthorized), "changed_engine_fields": sorted(changed_fields)}
    issues = attention_backend_issues(workspace, context)
    issues.extend(cache_implementation_issues(workspace))
    if issues:
        return {**result, "reason": "incompatible_or_ineffective_configuration", "issues": issues}
    native = [name for name in changed if Path(name).suffix in {".cu", ".cpp", ".c", ".h", ".hpp"}]
    if native and "native_kernel" not in allowed_surfaces:
        return {**result, "reason": "unauthorized_modification_surface",
                "surface": "native_kernel", "changed_files": native}
    if native and not context.get("engine_adapter", {}).get("build_command"):
        return {**result, "reason": "native_kernel_without_frozen_build", "changed_files": native}
    if not any(Path(name).suffix in {".py", ".json", ".cu", ".cpp", ".c", ".h", ".hpp"} for name in changed):
        return {**result, "reason": "no_runtime_change"}
    strategy = candidate_strategy(changed, changed_fields)
    result.update(strategy)
    # Families guide exploration; only an actually tested implementation/effect
    # can exclude a candidate. One cache or batch-size result cannot disprove
    # all other configurations, especially after an inconclusive measurement.
    for previous in contract.get("tested_candidates", []):
        if previous.get("fingerprint") == fingerprint:
            duplicate = {**result, "reason": "duplicate_change", "previous_iteration": previous["iteration"]}
            if previous.get("campaign_id"):
                duplicate.update(previous_campaign=previous["campaign_id"],
                                 previous_evidence_path=previous.get("evidence_path"),
                                 previous_decision=previous.get("decision"))
            return duplicate
    try:
        plan = complete_experiment_plan(plan, context)
    except (TypeError, ValueError) as error:
        return {**result, "reason": "invalid_bottleneck_metric", "error": str(error)}
    if not isinstance(plan, dict):
        return {**result, "reason": "missing_experiment_plan"}
    plan.update(strategy)
    if changed_fields & scheduler_fields:
        observation = context.get("performance_observation") or {}
        runtime = observation.get("runtime") or {}
        scheduler_capabilities = runtime.get("scheduler_capabilities") or {}
        if (candidate_config.get("request_scheduler") == "concurrent_requests"
                and "microbatch" in scheduler_capabilities.get("available", [])):
            return {**result, "reason": "unbatched_concurrent_scheduler",
                    "requested": "concurrent_requests", "recommended": "microbatch",
                    "evidence": {
                        "current_batching_mode": runtime.get("batching_mode", "none"),
                        "available_schedulers": scheduler_capabilities.get("available", []),
                    },
                    "explanation": ("This adapter exposes verified static microbatching. Concurrent "
                                    "request admission alone does not combine model.generate calls and "
                                    "must not be proposed as batching.")}
    if changed_fields & scheduler_fields and (contract.get("version", 1) >= 3):
        observation = context.get("performance_observation") or {}
        runtime = observation.get("runtime") or {}
        measurements = observation.get("measurements") or {}
        workload = context.get("workload") or {}
        workload_profiles = workload.get("profiles") or [workload]
        has_concurrent_profile = any(
            isinstance(profile.get("concurrency"), int) and profile["concurrency"] >= 2
            for profile in workload_profiles if isinstance(profile, dict)
        )
        metrics = sorted(name for name in measurements if name.startswith("diagnostic.concurrent.")
                         and name.endswith(".throughput"))
        if not has_concurrent_profile or not metrics:
            return {**result, "reason": "missing_scheduler_concurrency_evidence"}
        metric = metrics[0]
        measurement = measurements[metric]
        plan["bottleneck_reference"] = {
            "evidence_path": contract["observation_paths"][0], "metric": metric,
            "value": measurement.get("value"), "unit": measurement.get("unit"),
        }
        plan["evidence_paths"] = list(dict.fromkeys([
            contract["observation_paths"][0], *plan.get("evidence_paths", []),
        ]))
        plan["bottleneck_direction"] = "increase"
        plan["expected_bottleneck_change_pct"] = plan.get("expected_change_pct")
        plan["benefit_bound"] = _benefit_bound(
            metric, measurement, measurements, "increase", plan.get("expected_change_pct"))
        plan["mechanism_surface"] = "scheduler"
        plan["runtime_preconditions"] = [
            {"field": "execution_path", "equals": runtime["execution_path"]},
            {"field": "request_scheduler", "equals": runtime.get("request_scheduler", "serialized_lock")},
            {"field": "continuous_batching", "equals": runtime.get("continuous_batching", False)},
        ]
    for key in ("measured_bottleneck", "cause", "change", "validation"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            return {**result, "reason": "invalid_experiment_plan", "error": f"Missing {key}"}
    metric = plan.get("metric")
    if not isinstance(metric, str) or metric not in OBJECTIVES:
        return {**result, "reason": "invalid_experiment_plan",
                "error": "Unknown campaign objective metric",
                "allowed_metrics": sorted(OBJECTIVES),
                "hint": "Use bottleneck_metric for an exact observation measurements key."}
    expected = plan.get("expected_change_pct")
    if (plan.get("direction") not in ("increase", "decrease") or isinstance(expected, bool)
            or not isinstance(expected, (int, float)) or not math.isfinite(expected) or expected <= 0):
        return {**result, "reason": "invalid_experiment_plan", "error": "Predict a positive percentage and direction"}
    bound = plan.get("benefit_bound") or {}
    if bound.get("available") and bound.get("feasible") is False:
        return {**result, "reason": "implausible_expected_gain",
                "expected_change_pct": expected, "benefit_bound": bound}
    refs = plan.get("history_refs")
    if contract.get("version", 1) >= 3:
        known = set(contract.get("history_refs", []))
        invalid = (not isinstance(refs, list) or
                   any(not isinstance(ref, str) or ref not in known for ref in refs))
    else:
        known = set(contract.get("history_iterations", []))
        invalid = (not isinstance(refs, list) or
                   any(type(ref) is not int or ref not in known for ref in refs))
    if invalid:
        return {**result, "reason": "invalid_history_refs", "allowed_history_refs": sorted(known)[:50]}
    if known and not refs:
        return {**result, "reason": "missing_history_comparison"}
    evidence = plan.get("evidence_paths")
    if not isinstance(evidence, list) or not evidence:
        return {**result, "reason": "missing_measurement_evidence"}
    trusted = False
    trusted_records = {}
    try:
        for name in evidence:
            path = confined(Path(context["evidence_root"]), name)
            if not path.is_file():
                raise ValueError(f"Evidence is missing: {name}")
            # Only host-created diagnostics qualify as measured bottleneck evidence.
            if name in contract.get("observation_paths", []):
                record = read_json(path)
                bound = (isinstance(record, dict) and record.get("kind") == "host_inference_profile"
                            and record.get("source_hash") == source_hash(parent)
                            and record.get("profile_hash") == context["profile_hash"]
                            and record.get("workload") == context.get("workload")
                            and record.get("stable_ref") == context["stable_ref"])
                trusted |= bound
                if bound:
                    trusted_records[name] = record
    except (ValueError, TypeError, OSError) as error:
        return {**result, "reason": "invalid_measurement_evidence", "error": str(error)}
    if not trusted:
        return {**result, "reason": "unbound_measurement_evidence"}
    if contract.get("version", 1) >= 2:
        reference = plan.get("bottleneck_reference") or {}
        if (not isinstance(reference, dict) or not isinstance(reference.get("evidence_path"), str)
                or not isinstance(reference.get("metric"), str)):
            return {**result, "reason": "unbound_numeric_bottleneck"}
        record = trusted_records.get(reference.get("evidence_path")) or {}
        measurement = record.get("measurements", {}).get(reference.get("metric")) or {}
        if (not measurement or type(reference.get("value")) not in (int, float)
                or not math.isfinite(reference["value"]) or reference["value"] <= 0
                or reference.get("unit") != measurement.get("unit")
                or not math.isclose(reference["value"], measurement["value"], rel_tol=1e-6, abs_tol=1e-12)):
            return {**result, "reason": "unbound_numeric_bottleneck"}
        profiles = {p["id"] for p in context["workload"].get("profiles", [{"id": "default"}])}
        targets = plan.get("target_profile_ids")
        if not isinstance(targets, list) or not targets or any(not isinstance(p, str) or p not in profiles for p in targets):
            return {**result, "reason": "invalid_target_profiles"}
        if record.get("profile_id", "default") not in targets:
            return {**result, "reason": "bottleneck_profile_not_targeted"}
        preconditions = plan.get("runtime_preconditions")
        if not isinstance(preconditions, list) or not preconditions:
            return {**result, "reason": "missing_runtime_preconditions"}
        available_preconditions = _runtime_preconditions(record.get("runtime", {}))
        for condition in preconditions:
            current = record.get("runtime", {})
            field = condition.get("field") if isinstance(condition, dict) else None
            if not isinstance(field, str) or not field:
                return {**result, "reason": "invalid_runtime_precondition"}
            for key in field.split("."):
                current = current.get(key) if isinstance(current, dict) else None
            if current is None or current != condition.get("equals"):
                return {**result, "reason": "runtime_precondition_failed", "field": field,
                        "observed_value": current,
                        "available_runtime_preconditions": available_preconditions}
        prediction = plan.get("expected_bottleneck_change_pct")
        if (type(prediction) not in (int, float) or not math.isfinite(prediction) or prediction <= 0
                or plan.get("bottleneck_direction") not in ("increase", "decrease")
                or not isinstance(plan.get("expected_change_basis"), str) or not plan["expected_change_basis"].strip()):
            return {**result, "reason": "missing_bottleneck_prediction"}
    return {**result, "valid": True, "reason": "valid_candidate", "changed_files": changed,
            "experiment_plan": plan}
