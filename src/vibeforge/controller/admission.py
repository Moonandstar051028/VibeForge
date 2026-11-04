"""Deterministic admission policy; the agent cannot change these thresholds."""

from vibeforge.contracts.evaluation import Decision
from vibeforge.evaluation.statistics import paired_gain, percentile
from vibeforge.contracts.requirements import OBJECTIVES
import math


def decide(correctness, baseline, candidate, protocol):
    if not correctness["passed"]:
        return Decision("reject", "correctness_failed")
    if not baseline or len(baseline) != len(candidate):
        return Decision("invalid", "incomplete_measurements")
    if not all(row["valid"] for row in baseline + candidate):
        return Decision("invalid", "contaminated_measurements")
    objective = protocol.get("objective", "throughput")
    direction = OBJECTIVES[objective]
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
        values = [request.get(name, request.get("backend_metrics", {}).get(name))
                  for row in candidate for request in row["requests"]]
        if any(v is None for v in values) or not values or percentile(values, 0.95) > limit:
            return Decision("reject", f"protection_failed:{name}", **{k: interval[k] for k in ("gain", "lower", "upper")})
    outcome = "accept" if interval["lower"] > protocol["min_gain"] else "reject"
    reason = "measured_gain" if outcome == "accept" else (
        "performance_regression" if interval["upper"] < 0 else "no_significant_gain")
    return Decision(outcome, reason, **{key: interval[key] for key in ("gain", "lower", "upper")},
                    objective=objective, direction=direction)
