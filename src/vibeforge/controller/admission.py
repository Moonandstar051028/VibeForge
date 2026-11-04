"""Deterministic admission policy; the agent cannot change these thresholds."""

from vibeforge.contracts.evaluation import Decision
from vibeforge.evaluation.statistics import paired_gain, percentile


def decide(correctness, baseline, candidate, protocol):
    if not correctness["passed"]:
        return Decision("reject", "correctness_failed")
    if not baseline or len(baseline) != len(candidate):
        return Decision("invalid", "incomplete_measurements")
    if not all(row["valid"] for row in baseline + candidate):
        return Decision("invalid", "contaminated_measurements")
    interval = paired_gain([s["throughput"] for s in baseline], [s["throughput"] for s in candidate],
                           protocol.get("confidence", 0.95))
    for name, limit in protocol.get("absolute_limits", {}).items():
        values = [request.get(name) for row in candidate for request in row["requests"]]
        if any(v is None for v in values) or not values or percentile(values, 0.95) > limit:
            return Decision("reject", f"protection_failed:{name}", **{k: interval[k] for k in ("gain", "lower", "upper")})
    outcome = "accept" if interval["lower"] > protocol["min_gain"] else "reject"
    reason = "measured_gain" if outcome == "accept" else (
        "performance_regression" if interval["upper"] < 0 else "no_significant_gain")
    return Decision(outcome, reason, **{key: interval[key] for key in ("gain", "lower", "upper")})
