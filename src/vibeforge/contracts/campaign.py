"""Validated campaign configuration; JSON is the initial on-disk format."""

from pathlib import Path
import os
import math
from vibeforge.storage.atomic import digest, read_json


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
    if protocol.get("objective", "throughput") != "throughput":
        raise ValueError("The initial admission policy supports throughput as the primary objective")
    config["workload"].setdefault("max_new_tokens", 16)
    if not isinstance(config["workload"]["max_new_tokens"], int) or config["workload"]["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompts = config["workload"].get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError("workload.prompts must be a nonempty list of strings")
    if config["workload"].get("concurrency", 1) != 1:
        raise ValueError("This implementation measures single-request workloads; concurrency must be 1")
    if config["agent"].get("mode", "local") not in ("local", "scripted"):
        raise ValueError("agent.mode must be local or scripted")
    config["runtime_python"] = os.environ.get(
        "VIBEFORGE_RUNTIME_PYTHON", config.get("runtime_python", "python3"))
    config["inference_pythonpath"] = [str((project / path).resolve()) for path in config.get("inference_pythonpath", [])]
    config.setdefault("engine_template", "templates/engine/transformers")
    config["config_hash"] = digest({k: v for k, v in config.items() if k != "config_hash"})
    return config
