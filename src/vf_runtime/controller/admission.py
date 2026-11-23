"""Deterministic admission policy; the agent cannot change these thresholds."""

from vf_runtime.contracts.evaluation import Decision
from vf_runtime.evaluation.statistics import paired_gain, percentile
from vf_runtime.contracts.requirements import OBJECTIVES
import math


def decide(correctness, baseline, candidate, protocol):
    if not correctness["passed"]:
        return Decision("reject", "correctness_failed")
    if correctness.get('requires_quality') and correctness.get('quality_passed') is not True:
        return Decision('reject', 'independent_quality_unverified')
    if not baseline or len(baseline) != len(candidate):
        return Decision("invalid", "incomplete_measurements")
    if not all(row["valid"] for row in baseline + candidate):
        return Decision("invalid", "contaminated_measurements")
    objective = protocol.get("objective", "throughput")
    direction = OBJECTIVES[objective]
    required = {protocol.get("objective", "throughput"), *protocol.get("limits", {}),
                *protocol.get("max_metric_regressions", {})}
    if any("p95" in name or "p99" in name for name in required) or protocol.get("absolute_limits"):
        minimum = protocol.get("min_tail_requests", 100)
        for row in baseline + candidate:
            profiles = list(row.get("profile_records", {}).values()) or [row]
            if any(len(profile.get("requests", [])) < minimum for profile in profiles):
                return Decision("invalid", "insufficient_tail_samples")
    if any("profile_records" in row for row in baseline + candidate):
        names = set(baseline[0].get("profile_records", {}))
        if not names or any(set(row.get("profile_records", {})) != names for row in baseline + candidate):
            return Decision("invalid", "workload_profile_coverage_mismatch")
        if any(row.get("profile_weights") != baseline[0].get("profile_weights") for row in baseline + candidate):
            return Decision("invalid", "workload_profile_weights_mismatch")
        for name in sorted(names):
            profile_metric = objective if objective == "request_output_tokens_per_second_mean" else "throughput"
            def profile_value(row):
                record = row["profile_records"][name]
                return record.get('metrics', {}).get(profile_metric, record.get(profile_metric))
            a, b = [profile_value(row) for row in baseline], [profile_value(row) for row in candidate]
            if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in a + b):
                return Decision("invalid", "profile_metric_unavailable:" + name,
                                objective=objective, direction=direction)
            interval = paired_gain(a, b,
                                   protocol.get("confidence", .95))
            if interval["lower"] < -protocol.get("max_profile_regression", .03):
                outcome = "reject" if interval["upper"] < -protocol.get("max_profile_regression", .03) else "invalid"
                return Decision(outcome, "profile_regression:" + name if outcome == "reject" else "profile_regression_not_excluded:" + name,
                                objective=objective, direction=direction)
    # Protect latency independently of the throughput objective, per profile.
    for name, maximum in protocol.get('max_metric_regressions', {}).items():
        profiles = set(baseline[0].get('profile_records', {})) or {None}
        for profile in profiles:
            def value(row):
                record = row['profile_records'][profile] if profile else row
                return record.get('metrics', {}).get(name)
            a, b = [value(row) for row in baseline], [value(row) for row in candidate]
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in a + b):
                return Decision('invalid', 'protection_metric_unavailable:' + name)
            interval = paired_gain(a, b, protocol.get('confidence', .95), direction='minimize')
            if interval['lower'] < -maximum:
                reason = 'metric_regression:' if interval['upper'] < -maximum else 'metric_regression_not_excluded:'
                return Decision('reject' if interval['upper'] < -maximum else 'invalid', reason + str(profile) + ':' + name)
    def metric(row, name):
        return row.get("metrics", {}).get(name, row.get(name))
    for name, limit in protocol.get("limits", {}).items():
        values = [metric(row, name) for row in candidate]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v > limit for v in values):
            return Decision("reject", f"protection_failed:{name}", objective=objective, direction=direction)
    a, b = [metric(s, objective) for s in baseline], [metric(s, objective) for s in candidate]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in a + b):
        return Decision("invalid", "objective_metric_unavailable", objective=objective, direction=direction)
    interval = paired_gain(a, b, protocol.get("confidence", 0.95), direction=direction)
    for name, limit in protocol.get("absolute_limits", {}).items():
        values = [(request.get("host_observation") or {}).get(name)
                  for row in candidate for request in row["requests"]]
        if any(v is None for v in values) or not values or percentile(values, 0.95) > limit:
            return Decision("reject", f"protection_failed:{name}", **{k: interval[k] for k in ("gain", "lower", "upper")})
    outcome = "accept" if interval["lower"] > protocol["min_gain"] else "reject"
    reason = "measured_gain" if outcome == "accept" else (
        "performance_regression" if interval["upper"] < 0 else "no_significant_gain")
    return Decision(outcome, reason, **{key: interval[key] for key in ("gain", "lower", "upper")},
                    objective=objective, direction=direction)
