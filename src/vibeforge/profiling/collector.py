"""Compose a profile and a separate, transient deployment preflight."""

from vibeforge.contracts.profile import Profile
from vibeforge.profiling.model import inspect_model
from vibeforge.profiling.hardware import inspect_hardware
from vibeforge.profiling.environment import inspect_environment
from vibeforge.profiling.sampler import deployment_preflight


def collect(config):
    hardware = inspect_hardware()
    environment = inspect_environment(config["runtime_python"], config.get("inference_pythonpath", ()))
    if config["model"].get("fixture"):
        model = {"name": "toy-fixture", "fixture": True}
        preflight = {"passed": True, "reasons": [], "fixture": True}
    else:
        model = inspect_model(config["model"]["path"])
        preflight = deployment_preflight(model, hardware, config["model"], environment)
    return Profile(model, hardware, environment, config["workload"]).to_dict(), preflight
