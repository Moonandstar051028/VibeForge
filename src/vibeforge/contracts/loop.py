"""Frozen controls for telemetry, profiling, candidate probes, and refinement."""


def _object(config, name, legacy=None):
    value = config.get(name, legacy if legacy is not None else {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    config[name] = value
    return value


def _boolean(value, path):
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be boolean")


def _integer(value, path, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{path} must be an integer in [{minimum}, {maximum}]")


def normalize_loop_protocol(config, profile_ids):
    """Normalize phased loop controls while migrating the former observation block."""
    legacy = config.pop("observation", None)
    if legacy is not None and "profiling" in config:
        raise ValueError("Use profiling instead of configuring both profiling and legacy observation")

    telemetry = _object(config, "telemetry")
    unknown = set(telemetry) - {
        "enabled", "capture_agent_requests", "max_session_requests", "retention_seconds",
    }
    if unknown:
        raise ValueError(f"Unknown telemetry fields: {sorted(unknown)}")
    telemetry.setdefault("enabled", True)
    telemetry.setdefault("capture_agent_requests", True)
    telemetry.setdefault("max_session_requests", 256)
    telemetry.setdefault("retention_seconds", 3600)
    _boolean(telemetry["enabled"], "telemetry.enabled")
    _boolean(telemetry["capture_agent_requests"], "telemetry.capture_agent_requests")
    _integer(telemetry["max_session_requests"], "telemetry.max_session_requests", 1, 4096)
    _integer(telemetry["retention_seconds"], "telemetry.retention_seconds", 1, 86400)
    if telemetry["capture_agent_requests"] and not telemetry["enabled"]:
        raise ValueError("telemetry.capture_agent_requests requires telemetry.enabled")

    profiling = _object(config, "profiling", legacy=legacy)
    unknown = set(profiling) - {
        "enabled", "coverage", "max_new_tokens", "max_profiles_per_session",
        "max_matrix_profiles", "max_age_seconds", "timeout_seconds",
    }
    if unknown:
        raise ValueError(f"Unknown profiling fields: {sorted(unknown)}")
    profiling.setdefault("enabled", True)
    profiling.setdefault("coverage", "selected")
    profiling.setdefault("max_new_tokens", 128)
    profiling.setdefault("max_profiles_per_session", 2)
    profiling.setdefault("max_matrix_profiles", 8)
    profiling.setdefault("max_age_seconds", 600)
    profiling.setdefault("timeout_seconds", 300)
    _boolean(profiling["enabled"], "profiling.enabled")
    if profiling["coverage"] not in {"selected", "all"}:
        raise ValueError("profiling.coverage must be selected or all")
    _integer(profiling["max_new_tokens"], "profiling.max_new_tokens", 1, 128)
    _integer(profiling["max_profiles_per_session"], "profiling.max_profiles_per_session", 0, 16)
    _integer(profiling["max_matrix_profiles"], "profiling.max_matrix_profiles", 1, 64)
    _integer(profiling["max_age_seconds"], "profiling.max_age_seconds", 1, 86400)
    _integer(profiling["timeout_seconds"], "profiling.timeout_seconds", 1, 7200)
    if profiling["coverage"] == "all" and len(profile_ids) > profiling["max_matrix_profiles"]:
        raise ValueError("profiling.max_matrix_profiles does not cover the frozen workload matrix")

    probe = _object(config, "probe")
    unknown = set(probe) - {"enabled", "repeats", "warmup", "timeout_seconds", "max_attempts"}
    if unknown:
        raise ValueError(f"Unknown probe fields: {sorted(unknown)}")
    probe.setdefault("enabled", False)
    probe.setdefault("repeats", 1)
    probe.setdefault("warmup", 0)
    probe.setdefault("timeout_seconds", 300)
    probe.setdefault("max_attempts", 1)
    _boolean(probe["enabled"], "probe.enabled")
    _integer(probe["repeats"], "probe.repeats", 1, 3)
    _integer(probe["warmup"], "probe.warmup", 0, 10)
    _integer(probe["timeout_seconds"], "probe.timeout_seconds", 1, 7200)
    _integer(probe["max_attempts"], "probe.max_attempts", 1, 3)

    refinement = _object(config, "refinement")
    unknown = set(refinement) - {"enabled", "max_revisions", "timeout_seconds"}
    if unknown:
        raise ValueError(f"Unknown refinement fields: {sorted(unknown)}")
    refinement.setdefault("enabled", False)
    refinement.setdefault("max_revisions", 1)
    refinement.setdefault("timeout_seconds", 180)
    _boolean(refinement["enabled"], "refinement.enabled")
    _integer(refinement["max_revisions"], "refinement.max_revisions", 0, 2)
    _integer(refinement["timeout_seconds"], "refinement.timeout_seconds", 1, 3600)
    if refinement["enabled"] and not probe["enabled"]:
        raise ValueError("refinement.enabled requires probe.enabled")
    if refinement["enabled"] and refinement["max_revisions"] < 1:
        raise ValueError("refinement.enabled requires at least one revision")
    if refinement["enabled"] and probe["max_attempts"] < refinement["max_revisions"] + 1:
        raise ValueError("probe.max_attempts must cover the initial probe and every configured revision")

    config["workload_roles"] = {
        "deployment": {"profile_ids": list(profile_ids), "admission": True},
        "agent_self_hosting": {
            "enabled": telemetry["enabled"] and telemetry["capture_agent_requests"],
            "source": "live_agent_requests",
            "admission": False,
        },
    }
    return {
        "telemetry": telemetry,
        "profiling": profiling,
        "probe": probe,
        "refinement": refinement,
    }
