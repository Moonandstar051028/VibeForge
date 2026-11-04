"""Compose a profile and a separate, transient deployment preflight."""

from vibeforge.contracts.profile import Profile
from vibeforge.profiling.model import inspect_model
from vibeforge.profiling.hardware import inspect_hardware
from vibeforge.profiling.environment import inspect_environment
from vibeforge.profiling.sampler import deployment_preflight
from vibeforge.contracts.precision import normalize_precision
from copy import deepcopy
from vibeforge.engine.adapters import compatibility_errors


def collect(config):
    hardware = inspect_hardware()
    environment = inspect_environment(config["runtime_python"], config.get("inference_pythonpath", ()))
    if config["model"].get("fixture"):
        model = {"name": "toy-fixture", "fixture": True}
        preflight = {"passed": True, "reasons": [], "fixture": True}
    else:
        model = inspect_model(config["model"]["path"], config.get("engine", {}).get("model_format", "safetensors"))
        target = {**config["model"], "precision": config.get("precision"), "engine": config.get("engine", {}),
                  "hardware_requirements": config.get("hardware", {})}
        preflight = deployment_preflight(model, hardware, target, environment)
    if config.get("engine"):
        issues = compatibility_errors(config, normalize_precision(config.get("precision"), config["model"]))
        if config["engine"]["adapter"] == "transformers" and model.get("prequantized"):
            issues.append("prequantized_checkpoint_requires_explicit_python_adapter")
        preflight["reasons"].extend(issues)
        preflight["passed"] = not preflight["reasons"]
    stable_environment = deepcopy(environment)
    for device in stable_environment.get("inference", {}).get("runtime", {}).get("devices", []):
        device.pop("free_bytes", None)
    execution = {"precision": normalize_precision(config.get("precision"), config["model"]),
                 "device": config["model"].get("device", "cpu"), "engine": config.get("engine", {}),
                 "requirements": config.get("requirements", {}), "evaluation": config["evaluation"]}
    return Profile(model, hardware, stable_environment, config["workload"], execution).to_dict(), preflight
