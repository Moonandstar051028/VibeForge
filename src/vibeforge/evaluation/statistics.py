"""Paired run-level bootstrap intervals, not token-level pseudo-replication."""

import math
import random
import statistics


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        raise ValueError("Empty sample")
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def paired_gain(baseline, candidate, confidence=0.95, resamples=2000, seed=17, direction="maximize"):
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError("Paired measurements require matching run groups and at least two pairs")
    if not all(math.isfinite(x) and x > 0 for x in baseline + candidate):
        raise ValueError("Throughput samples must be finite and positive")
    if direction not in ("maximize", "minimize"):
        raise ValueError("Unknown objective direction")
    gains = [(c / b - 1) if direction == "maximize" else (1 - c / b) for b, c in zip(baseline, candidate)]
    randomizer = random.Random(seed)
    means = [statistics.mean(randomizer.choices(gains, k=len(gains))) for _ in range(resamples)]
    tail = (1 - confidence) / 2
    return {"gain": statistics.mean(gains), "lower": percentile(means, tail),
            "upper": percentile(means, 1 - tail), "confidence": confidence,
            "pairs": len(gains), "method": "paired_run_bootstrap", "resamples": resamples}
