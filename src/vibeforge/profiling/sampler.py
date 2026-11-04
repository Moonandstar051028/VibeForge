"""Transient device observations and deployment preflight."""

import os
import time
from pathlib import Path
from vibeforge.profiling.hardware import memory_info, gpu_processes, query


def sample():
    temperatures = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temperatures.append(int(path.read_text().strip()) / 1000)
        except (OSError, ValueError):
            pass
    return {
        "monotonic_time": time.monotonic(), **memory_info(),
        "temperature_c": max(temperatures) if temperatures else None,
        "gpu_processes": gpu_processes(),
        "gpu_state": query(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw", "--format=csv,noheader"]),
        "load_average": list(os.getloadavg()),
    }


def deployment_preflight(model, hardware, target, runtime_environment):
    available = memory_info()["available_bytes"]
    quantization = target.get("quantization")
    factor = 0.35 if quantization == "bitsandbytes-4bit" else 1.08
    required = int(model["weight_bytes"] * factor + target.get("reserve_gib", 6) * 1024 ** 3)
    packages = runtime_environment.get("inference", {}).get("packages", {})
    missing = [name for name in ("torch", "transformers", "safetensors", "accelerate") if not packages.get(name)]
    if quantization and quantization != "bitsandbytes-4bit":
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    if quantization:
        missing.extend(name for name in ("accelerate", "bitsandbytes") if not packages.get(name))
    reasons = []
    if available < required:
        reasons.append("insufficient_memory")
    if missing:
        reasons.append("missing_inference_dependencies")
    if target.get("device", "cuda") == "cuda" and not hardware.get("gpu"):
        reasons.append("cuda_device_unavailable")
    return {
        "passed": not reasons, "reasons": reasons, "missing_packages": missing,
        "weight_bytes": model["weight_bytes"], "required_bytes_estimate": required,
        "available_bytes": available, "quantization": quantization,
        "other_gpu_processes": gpu_processes(),
        "estimate_note": "BF16 weights plus headroom; 4-bit estimate is not an allocation guarantee",
    }
