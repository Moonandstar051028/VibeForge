"""Validated campaign configuration; JSON is the initial on-disk format."""

from pathlib import Path
import os
import math
import hashlib
from vibeforge.storage.atomic import digest, read_json
from vibeforge.contracts.precision import normalize_precision
from vibeforge.contracts.requirements import normalize_requirements
from vibeforge.engine.adapters import normalize_engine
from vibeforge.contracts.workload import normalize_workload, workload_profiles
from vibeforge.contracts.loop import normalize_loop_protocol


def load_config(path, project):
    path, project = Path(path).resolve(), Path(project).resolve()
    config = read_json(path)
    knowledge = config.setdefault("knowledge", {})
    if not isinstance(knowledge, dict):
        raise ValueError("knowledge must be an object")
    scope = knowledge.setdefault("scope", "project")
    if scope not in ("project", "current_campaign"):
        raise ValueError("knowledge.scope must be project or current_campaign")
    for section in ("model", "hardware", "workload", "evaluation", "agent"):
        value = config.get(section)
        if isinstance(value, str):
            value = read_json(project / value)
        if not isinstance(value, dict):
            raise ValueError(f"{section} must be an object or project-relative JSON reference")
        config[section] = value
    model_path = os.environ.get("VIBEFORGE_MODEL_PATH", config["model"].get("path"))
    if model_path:
        model_path = Path(model_path)
        config["model"]["path"] = str((project / model_path).resolve() if not model_path.is_absolute() else model_path)
    max_context = config["model"].get("max_context")
    if (max_context is not None
            and (not isinstance(max_context, int) or isinstance(max_context, bool) or max_context <= 0)):
        raise ValueError("model.max_context must be a positive integer when provided")
    model_config_path = Path(config["model"].get("path", "")) / "config.json"
    native_context = None
    if model_config_path.is_file():
        model_config = read_json(model_config_path)
        text_config = model_config.get("text_config", model_config)
        native_context = text_config.get("max_position_embeddings")
        if isinstance(native_context, int) and native_context > 0:
            config["model"]["native_max_context"] = native_context
            if max_context is not None and max_context > native_context:
                raise ValueError(
                    f"model.max_context ({max_context}) exceeds the model's native limit ({native_context})")
        else:
            native_context = None
    if max_context is None:
        max_context = native_context or 4096
        config["model"]["max_context"] = max_context
        config["model"]["context_limit_source"] = "model_config" if native_context else "fallback_4096"
    else:
        config["model"]["context_limit_source"] = "campaign"
    config["model"].setdefault("device", config["hardware"].get("device", "cpu"))
    for section in ("precision", "requirements", "optimization", "engine"):
        if isinstance(config.get(section), str):
            config[section] = read_json(project / config[section])
    config["precision"] = normalize_precision(config.get("precision"), config["model"])
    policy = config.setdefault("optimization", {})
    if not isinstance(policy, dict):
        raise ValueError("optimization must be an object")
    policy.setdefault("allow_precision_changes", False)
    if not isinstance(policy["allow_precision_changes"], bool):
        raise ValueError("optimization.allow_precision_changes must be boolean")
    choices = policy.get("allowed_precisions", [])
    if not isinstance(choices, list):
        raise ValueError("optimization.allowed_precisions must be a list")
    policy["allowed_precisions"] = [normalize_precision(read_json(project / v) if isinstance(v, str) else v)
                                    for v in choices]
    allowed_surfaces = policy.setdefault("allowed_surfaces", [
        "runtime_config", "execution_adapter", "cache_policy", "scheduler",
    ])
    known_surfaces = {
        "runtime_config", "execution_adapter", "cache_policy", "scheduler", "native_kernel",
    }
    if (not isinstance(allowed_surfaces, list) or not allowed_surfaces
            or any(not isinstance(value, str) or value not in known_surfaces
                   for value in allowed_surfaces)
            or len(set(allowed_surfaces)) != len(allowed_surfaces)):
        raise ValueError("optimization.allowed_surfaces must be a unique nonempty list of known surfaces")
    if policy["allow_precision_changes"]:
        if config["precision"] not in policy["allowed_precisions"]:
            policy["allowed_precisions"].insert(0, config["precision"])
        if len(policy["allowed_precisions"]) < 2:
            raise ValueError("Precision search needs at least two allowed_precisions")
    budget = config.setdefault("budget", {})
    for key, default in (("max_iterations", 2), ("max_seconds", 3600), ("agent_seconds", 180), ("max_failures", 3),
                         ("max_proposals", budget.get("max_iterations", 2) * 3), ("max_stalled_attempts", 3),
                         ("max_candidate_failures", 5), ("max_analysis_attempts", 2)):
        budget.setdefault(key, default)
        if not isinstance(budget[key], (int, float)) or isinstance(budget[key], bool) or not math.isfinite(budget[key]) or budget[key] <= 0:
            raise ValueError(f"budget.{key} must be positive")
        if key in ("max_iterations", "max_failures", "max_proposals", "max_stalled_attempts", "max_candidate_failures", "max_analysis_attempts") and not isinstance(budget[key], int):
            raise ValueError(f"budget.{key} must be an integer")
    if budget["max_analysis_attempts"] > 3:
        raise ValueError("budget.max_analysis_attempts must not exceed 3")
    protocol = config["evaluation"]
    protocol.setdefault("measurement_protocol_version", 2)
    if protocol["measurement_protocol_version"] != 2:
        raise ValueError("Create a new campaign with measurement protocol version 2")
    for key, default in (("repeats", 5), ("warmup", 1), ("min_gain", 0.02), ("confidence", 0.95), ("max_retests", 1)):
        protocol.setdefault(key, default)
    if protocol["repeats"] < 2 or not 0 < protocol["confidence"] < 1:
        raise ValueError("At least two repeats and a confidence level inside (0, 1) are required")
    if protocol["min_gain"] < 0 or protocol["warmup"] < 0 or protocol["max_retests"] < 0:
        raise ValueError("Invalid evaluation thresholds")
    for key in ("repeats", "warmup", "max_retests"):
        if not isinstance(protocol[key], int) or isinstance(protocol[key], bool):
            raise ValueError(f"evaluation.{key} must be an integer")
    if not all(math.isfinite(protocol[key]) for key in ("confidence", "min_gain")):
        raise ValueError("Evaluation thresholds must be finite")
    normalize_requirements(config)
    protocol.setdefault("correctness_mode", "strict")
    if protocol["correctness_mode"] not in ("strict", "approximate"):
        raise ValueError("evaluation.correctness_mode must be strict or approximate")
    if protocol["correctness_mode"] == "strict" and protocol.get("exact_tokens") is False:
        raise ValueError("Disable exact tokens only with approximate mode and a quality evaluator")
    protocol.setdefault("min_token_agreement", 0.9)
    if not isinstance(protocol["min_token_agreement"], (int, float)) or not 0 < protocol["min_token_agreement"] <= 1:
        raise ValueError("min_token_agreement must be in (0, 1]")
    protocol.setdefault("logit_samples", 64)
    if not isinstance(protocol["logit_samples"], int) or not 1 <= protocol["logit_samples"] <= 4096:
        raise ValueError("logit_samples must be in [1, 4096]")
    protocol.setdefault("reference_repeats", 3)
    if type(protocol["reference_repeats"]) is not int or not 2 <= protocol["reference_repeats"] <= 10:
        raise ValueError("evaluation.reference_repeats must be an integer in [2, 10]")
    protocol.setdefault("min_reference_consensus", 2 / 3)
    consensus = protocol["min_reference_consensus"]
    if isinstance(consensus, bool) or not isinstance(consensus, (int, float)) or not math.isfinite(consensus) or not 0 < consensus <= 1:
        raise ValueError("evaluation.min_reference_consensus must be in (0, 1]")
    for key in ("atol", "rtol", "reference_atol", "reference_rtol", "max_normalized_rms"):
        if key in protocol and (not isinstance(protocol[key], (int, float)) or
                                isinstance(protocol[key], bool) or not math.isfinite(protocol[key]) or protocol[key] < 0):
            raise ValueError(f"evaluation.{key} must be finite and nonnegative")
    quality = protocol.get("quality")
    if protocol["correctness_mode"] == "approximate" or policy["allow_precision_changes"]:
        if not quality:
            raise ValueError("Approximation/precision search requires an independent quality evaluator")
        if policy["allow_precision_changes"] and protocol["correctness_mode"] != "approximate":
            raise ValueError("Precision search must explicitly use approximate correctness mode")
    if quality:
        if not isinstance(quality, dict) or not isinstance(quality.get("command"), list) or not quality["command"]:
            raise ValueError("evaluation.quality.command must be a nonempty argv list")
        if not all(isinstance(v, str) and v for v in quality["command"]):
            raise ValueError("Quality command arguments must be strings")
        if not quality.get("min_scores"):
            raise ValueError("Quality evaluator requires explicit min_scores")
        if policy["allow_precision_changes"] and not quality.get("max_regressions"):
            raise ValueError("Precision search requires max_regressions against the frozen initial quality")
        for key in ("min_scores", "max_regressions"):
            values = quality.get(key, {})
            if not isinstance(values, dict) or any(not isinstance(v, (int, float)) or isinstance(v, bool) or
                                                 not math.isfinite(v) or (key == "max_regressions" and v < 0)
                                                 for v in values.values()):
                raise ValueError(f"Quality {key} must contain finite thresholds")
        quality.setdefault("timeout_seconds", 300)
        if not isinstance(quality["timeout_seconds"], (int, float)) or not 0 < quality["timeout_seconds"] <= 7200:
            raise ValueError("Quality timeout_seconds must be in (0, 7200]")
        files = quality.get("files", [])
        if not isinstance(files, list) or not files or not all(isinstance(v, str) for v in files):
            raise ValueError("Quality evaluator must declare files (script, dataset/config) for integrity checks")
        quality.setdefault("repeats", 3)
        if type(quality["repeats"]) is not int or not 1 <= quality["repeats"] <= 10:
            raise ValueError("evaluation.quality.repeats must be an integer in [1, 10]")
        quality["files"] = [str((project / path).resolve()) for path in files]
        quality["file_hashes"] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in quality["files"]}
    if not config["model"].get("fixture") and not protocol.get("reference_path"):
        if not quality:
            raise ValueError("A self-generated regression reference requires an independent quality evaluator")
        if not quality.get("max_regressions"):
            raise ValueError("A self-generated regression reference requires quality.max_regressions")
    if protocol.get("reference_path"):
        reference_path = (project / protocol["reference_path"]).resolve()
        protocol["reference_path"] = str(reference_path)
        protocol["reference_file_hash"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    normalize_workload(config["workload"])
    normalize_loop_protocol(config, [profile["id"] for profile in workload_profiles(config["workload"])])
    config["workload"].setdefault("max_new_tokens", 16)
    if not isinstance(config["workload"]["max_new_tokens"], int) or config["workload"]["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompts = config["workload"].get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError("workload.prompts must be a nonempty list of strings")
    concurrency = config["workload"].setdefault("concurrency", 1)
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or not 1 <= concurrency <= 64:
        raise ValueError("workload.concurrency must be an integer in [1, 64]")
    protocol["concurrency"] = max(p["concurrency"] for p in workload_profiles(config["workload"]))
    tolerance = protocol.setdefault("max_profile_regression", 0.03)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 <= tolerance < 1:
        raise ValueError("evaluation.max_profile_regression must be in [0, 1)")
    protocol["device_backend"] = config["model"]["device"].split(":", 1)[0]
    device = config["model"]["device"]
    if device != "cpu" and not (device.split(":", 1)[0] in ("cuda", "rocm") and (
        ":" not in device or device.split(":", 1)[1].isdigit())):
        raise ValueError("model.device must be cpu, cuda[:index] or rocm[:index]")
    interval = protocol.setdefault("resource_sample_interval", 0.05)
    if not isinstance(interval, (int, float)) or isinstance(interval, bool) or not 0.005 <= interval <= 1:
        raise ValueError("resource_sample_interval must be in [0.005, 1] seconds")
    device_interval = protocol.setdefault("device_sample_interval", 1.0)
    if (not isinstance(device_interval, (int, float)) or isinstance(device_interval, bool)
            or not 0.05 <= device_interval <= 10):
        raise ValueError("device_sample_interval must be in [0.05, 10] seconds")
    min_tail_requests = protocol.setdefault("min_tail_requests", 100)
    if (not isinstance(min_tail_requests, int) or isinstance(min_tail_requests, bool)
            or not 20 <= min_tail_requests <= 100000):
        raise ValueError("min_tail_requests must be an integer in [20, 100000]")
    require_device_telemetry = protocol.setdefault("require_continuous_device_telemetry", False)
    if not isinstance(require_device_telemetry, bool):
        raise ValueError("require_continuous_device_telemetry must be boolean")
    if require_device_telemetry and protocol["device_backend"] != "cuda":
        raise ValueError("require_continuous_device_telemetry currently requires model.device=cuda[:index]")
    if config["agent"].get("mode", "local") not in ("local", "scripted"):
        raise ValueError("agent.mode must be local or scripted (test fixture)")
    if config["agent"].get("mode", "local") == "local":
        config["agent"].setdefault("framework", "qwen-agent")
        if config["agent"]["framework"] not in ("qwen-agent", "codex"):
            raise ValueError("agent.framework must be qwen-agent or codex")
        config["agent"].setdefault("max_turns", 6)
        if not isinstance(config["agent"]["max_turns"], int) or not 1 <= config["agent"]["max_turns"] <= 20:
            raise ValueError("agent.max_turns must be an integer from 1 to 20")
        config["agent"].setdefault("max_new_tokens", 512)
        if not isinstance(config["agent"]["max_new_tokens"], int) or config["agent"]["max_new_tokens"] <= 0:
            raise ValueError("agent.max_new_tokens must be a positive integer")
        if config["agent"]["max_new_tokens"] >= config["model"]["max_context"]:
            raise ValueError("agent.max_new_tokens must be smaller than model.max_context")
        thinking = config["agent"].setdefault("thinking", {"enabled": False})
        if not isinstance(thinking, dict):
            raise ValueError("agent.thinking must be an object")
        thinking.setdefault("enabled", False)
        if not isinstance(thinking["enabled"], bool):
            raise ValueError("agent.thinking.enabled must be boolean")
        defaults = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
        unknown = set(thinking) - {"enabled", *defaults}
        if unknown:
            raise ValueError(f"Unknown agent.thinking fields: {sorted(unknown)}")
        for key, default in defaults.items():
            thinking.setdefault(key, default)
        if (isinstance(thinking["temperature"], bool)
                or not isinstance(thinking["temperature"], (int, float))
                or not math.isfinite(thinking["temperature"]) or thinking["temperature"] <= 0):
            raise ValueError("agent.thinking.temperature must be finite and greater than zero")
        for key in ("top_p", "min_p"):
            value = thinking[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"agent.thinking.{key} must be in [0, 1]")
        if (not isinstance(thinking["top_k"], int) or isinstance(thinking["top_k"], bool)
                or thinking["top_k"] < 0):
            raise ValueError("agent.thinking.top_k must be a nonnegative integer")
        if config["agent"]["framework"] == "codex":
            prompt = config["agent"].setdefault(
                "prompt", "templates/prompts/codex-implementation.md")
            if prompt == "templates/prompts/implementation.md":
                raise ValueError(
                    "Codex requires templates/prompts/codex-implementation.md or another Codex-specific prompt")
            executable = config["agent"].setdefault("executable", "codex")
            if not isinstance(executable, str) or not executable:
                raise ValueError("agent.executable must be a nonempty string")
            if "compact_at_tokens" in config["agent"]:
                raise ValueError(
                    "agent.compact_at_tokens is unsupported; Codex owns automatic context compaction")
            max_passes = config["agent"].setdefault("max_passes", 3)
            if (not isinstance(max_passes, int) or isinstance(max_passes, bool)
                    or not 1 <= max_passes <= 5):
                raise ValueError("agent.max_passes must be an integer from 1 to 5")
            codex_model = config["agent"].setdefault("model", config["model"].get("name", "vibeforge-local"))
            if not isinstance(codex_model, str) or not codex_model:
                raise ValueError("agent.model must be a nonempty string")
    config["runtime_python"] = os.environ.get(
        "VIBEFORGE_RUNTIME_PYTHON", config.get("runtime_python", "python3"))
    config["inference_pythonpath"] = [str((project / path).resolve()) for path in config.get("inference_pythonpath", [])]
    normalize_engine(config, project)
    if "native_kernel" in allowed_surfaces and not config["engine"].get("build_command"):
        raise ValueError("native_kernel optimization requires a frozen engine.build_command")
    config["schema_version"] = 4
    config["config_hash"] = digest({k: v for k, v in config.items() if k != "config_hash"})
    return config
