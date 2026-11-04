"""Transient device observations and deployment preflight."""

import os
import time
from pathlib import Path
from vibeforge.profiling.hardware import memory_info, gpu_processes, query
from vibeforge.contracts.precision import normalize_precision, FLOAT_DTYPES


def estimate_memory(model, target):
    precision = normalize_precision(target.get("precision"), target)
    if target.get("estimated_weight_bytes") is not None:
        weights = target["estimated_weight_bytes"]
        if not isinstance(weights, int) or isinstance(weights, bool) or weights <= 0:
            raise ValueError("estimated_weight_bytes must be a positive integer")
        method = "user_supplied_runtime_weight_estimate"
    elif model.get("prequantized"):
        weights, method = model["weight_bytes"], "prepacked_storage_bytes"
    elif model.get("dtype_elements"):
        elements = model["dtype_elements"]
        floating = sum(count for dtype, count in elements.items() if dtype.startswith(("F", "BF")))
        fixed = sum(count for dtype, count in model.get("dtype_bytes", {}).items() if not dtype.startswith(("F", "BF")))
        q = precision["quantization"]
        if q.startswith("custom:"):
            raise ValueError("Custom quantization needs prequantized artifacts or estimated_weight_bytes")
        per_element = 0.66 if q in ("bnb-nf4", "bnb-fp4") else (1.3 if q == "bnb-int8" else FLOAT_DTYPES[precision["weight_dtype"]])
        weights, method = int(floating * per_element + fixed), "tensor_dtype_and_target_precision"
    else:
        # Old profiles can still be read, but new inspectors provide dtype counts.
        weights = int(model["weight_bytes"] * (0.33 if precision["quantization"] in ("bnb-nf4", "bnb-fp4") else 1))
        method = "legacy_storage_estimate"
    reserve = target.get("reserve_gib", 6) * 1024 ** 3
    return {"runtime_weight_bytes_estimate": int(weights),
            "required_bytes_estimate": int(weights * 1.08 + reserve), "estimate_method": method,
            "estimate_note": "Conservative weight/headroom estimate; KV, activation and compiler peaks require measured limits"}


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
    precision = normalize_precision(target.get("precision"), target)
    quantization = precision["quantization"]
    estimate = estimate_memory(model, target)
    required = estimate["required_bytes_estimate"]
    engine = target.get("engine", {})
    packages = runtime_environment.get("inference", {}).get("packages", {})
    required_packages = engine.get("required_packages", ["torch", "transformers", "safetensors", "accelerate"])
    missing = [name for name in required_packages if not packages.get(name.lower())]
    if quantization.startswith("bnb-"):
        missing.extend(name for name in ("accelerate", "bitsandbytes") if not packages.get(name))
    reasons = []
    device = target.get("device", "cuda")
    backend, _, index = device.partition(":")
    if engine.get("adapter") == "transformers" and backend in ("cuda", "rocm") and not packages.get("accelerate"):
        missing.append("accelerate")
    runtime = runtime_environment.get("inference", {}).get("runtime", {})
    devices = runtime.get("devices", [])
    if backend in ("cuda", "rocm"):
        selected = next((d for d in devices if d["index"] == int(index or 0)), None)
        if runtime and (not selected or runtime.get("backend") != backend):
            reasons.append(f"{backend}_device_unavailable")
        elif not runtime and not hardware.get("gpu"):
            reasons.append(f"{backend}_device_unavailable")
        if selected:
            available = min(available, selected["free_bytes"])
            if precision["compute_dtype"] == "bfloat16" and not selected.get("bf16_supported"):
                reasons.append("device_bfloat16_unsupported")
    elif backend != "cpu":
        reasons.append(f"device_backend_not_probed:{backend}")
    if available < required:
        reasons.append("insufficient_memory")
    if missing:
        reasons.append("missing_inference_dependencies")
    constraints = target.get("hardware_requirements", {})
    if constraints.get("architecture") and constraints["architecture"] != hardware.get("architecture"):
        reasons.append("hardware_architecture_mismatch")
    if constraints.get("name_contains") and constraints["name_contains"] not in str(hardware.get("gpu", "")):
        reasons.append("hardware_name_mismatch")
    return {
        "passed": not reasons, "reasons": reasons, "missing_packages": missing,
        "weight_bytes": model["weight_bytes"], **estimate,
        "available_bytes": available, "quantization": quantization, "precision": precision,
        "other_gpu_processes": gpu_processes(),
    }
