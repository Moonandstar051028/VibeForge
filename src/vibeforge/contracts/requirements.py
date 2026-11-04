"""User objectives and resource limits accepted by the independent evaluator."""

import math


OBJECTIVES = {
    "throughput": "maximize",
    "latency_p95_seconds": "minimize",
    "peak_rss_bytes": "minimize",
    "startup_seconds": "minimize",
    "source_bytes": "minimize",
}
CAPABILITIES = {"generate", "logits", "chat", "chat_completion"}


def normalize_requirements(config):
    value = config.setdefault("requirements", {})
    if not isinstance(value, dict):
        raise ValueError("requirements must be an object")
    unknown = set(value) - {"objective", "limits", "description", "capabilities"}
    if unknown:
        raise ValueError(f"Unknown requirement fields: {sorted(unknown)}")
    objective = value.setdefault("objective", config["evaluation"].get("objective", "throughput"))
    if objective not in OBJECTIVES:
        raise ValueError(f"Unsupported objective: {objective}; choose {sorted(OBJECTIVES)}")
    config["evaluation"]["objective"] = objective
    limits = value.setdefault("limits", {})
    if not isinstance(limits, dict):
        raise ValueError("requirements.limits must be an object")
    for name, limit in limits.items():
        if name not in OBJECTIVES or name == "throughput":
            raise ValueError(f"Unsupported upper-bound metric: {name}")
        if not isinstance(limit, (int, float)) or isinstance(limit, bool) or not math.isfinite(limit) or limit <= 0:
            raise ValueError(f"requirements.limits.{name} must be positive and finite")
    capabilities = value.setdefault("capabilities", ["generate", "logits", "chat_completion"])
    if not isinstance(capabilities, list) or not all(c in CAPABILITIES for c in capabilities):
        raise ValueError("requirements.capabilities contains an unsupported interface")
    # These interfaces are required by differential evaluation and the self-hosted agent.
    value["capabilities"] = sorted(set(capabilities) | {"generate", "logits", "chat_completion"})
    config["evaluation"]["limits"] = dict(limits)
    return value
