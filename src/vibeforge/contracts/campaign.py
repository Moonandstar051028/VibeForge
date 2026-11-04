"""Validated campaign configuration; JSON is the initial on-disk format."""

from pathlib import Path
import os
import math
import hashlib
from vibeforge.storage.atomic import digest, read_json
from vibeforge.contracts.precision import normalize_precision
from vibeforge.contracts.requirements import normalize_requirements
from vibeforge.engine.adapters import normalize_engine


def load_config(path, project):
    path, project = Path(path).resolve(), Path(project).resolve()
    config = read_json(path)
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
    if policy["allow_precision_changes"]:
        if config["precision"] not in policy["allowed_precisions"]:
            policy["allowed_precisions"].insert(0, config["precision"])
        if len(policy["allowed_precisions"]) < 2:
            raise ValueError("Precision search needs at least two allowed_precisions")
    budget = config.setdefault("budget", {})
    for key, default in (("max_iterations", 2), ("max_seconds", 3600), ("agent_seconds", 180), ("max_failures", 3)):
        budget.setdefault(key, default)
        if not isinstance(budget[key], (int, float)) or isinstance(budget[key], bool) or not math.isfinite(budget[key]) or budget[key] <= 0:
            raise ValueError(f"budget.{key} must be positive")
        if key in ("max_iterations", "max_failures") and not isinstance(budget[key], int):
            raise ValueError(f"budget.{key} must be an integer")
    protocol = config["evaluation"]
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
    for key in ("atol", "rtol", "max_normalized_rms"):
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
        quality["files"] = [str((project / path).resolve()) for path in files]
        quality["file_hashes"] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in quality["files"]}
    if protocol.get("reference_path"):
        reference_path = (project / protocol["reference_path"]).resolve()
        protocol["reference_path"] = str(reference_path)
        protocol["reference_file_hash"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    config["workload"].setdefault("max_new_tokens", 16)
    if not isinstance(config["workload"]["max_new_tokens"], int) or config["workload"]["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompts = config["workload"].get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError("workload.prompts must be a nonempty list of strings")
    concurrency = config["workload"].setdefault("concurrency", 1)
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or not 1 <= concurrency <= 64:
        raise ValueError("workload.concurrency must be an integer in [1, 64]")
    protocol["concurrency"] = concurrency
    protocol["device_backend"] = config["model"]["device"].split(":", 1)[0]
    device = config["model"]["device"]
    if device != "cpu" and not (device.split(":", 1)[0] in ("cuda", "rocm") and (
        ":" not in device or device.split(":", 1)[1].isdigit())):
        raise ValueError("model.device must be cpu, cuda[:index] or rocm[:index]")
    interval = protocol.setdefault("resource_sample_interval", 0.05)
    if not isinstance(interval, (int, float)) or isinstance(interval, bool) or not 0.005 <= interval <= 1:
        raise ValueError("resource_sample_interval must be in [0.005, 1] seconds")
    if config["agent"].get("mode", "local") not in ("local", "scripted"):
        raise ValueError("agent.mode must be local (Qwen-Agent) or scripted (test fixture)")
    if config["agent"].get("mode", "local") == "local":
        config["agent"].setdefault("framework", "qwen-agent")
        if config["agent"]["framework"] != "qwen-agent":
            raise ValueError("agent.framework must be qwen-agent")
        config["agent"].setdefault("max_turns", 6)
        if not isinstance(config["agent"]["max_turns"], int) or not 1 <= config["agent"]["max_turns"] <= 20:
            raise ValueError("agent.max_turns must be an integer from 1 to 20")
        config["agent"].setdefault("max_new_tokens", 512)
        if not isinstance(config["agent"]["max_new_tokens"], int) or config["agent"]["max_new_tokens"] <= 0:
            raise ValueError("agent.max_new_tokens must be a positive integer")
    config["runtime_python"] = os.environ.get(
        "VIBEFORGE_RUNTIME_PYTHON", config.get("runtime_python", "python3"))
    config["inference_pythonpath"] = [str((project / path).resolve()) for path in config.get("inference_pythonpath", [])]
    normalize_engine(config, project)
    config["schema_version"] = 2
    config["config_hash"] = digest({k: v for k, v in config.items() if k != "config_hash"})
    return config
