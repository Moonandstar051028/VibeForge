"""Describe user-supplied initial engines without importing inference libraries."""

from pathlib import Path
from vibeforge.contracts.precision import QUANTIZATION
from vibeforge.versioning.artifacts import source_hash


def capabilities():
    return {
        "schema_version": 2,
        "adapters": ["transformers", "python"],
        "model_formats": ["safetensors", "gguf", "manifest"],
        "devices": ["cpu", "cuda[:index]", "rocm[:index]"],
        "compute_dtypes": ["float32", "float16", "bfloat16"],
        "quantization": sorted(QUANTIZATION) + ["custom:<name> (explicit Python adapter)"],
        "objectives": ["throughput", "latency_p95_seconds", "peak_rss_bytes", "startup_seconds", "source_bytes"],
        "concurrency": {"min": 1, "max": 64,
                        "transport": "candidate-selectable serialized_lock, concurrent_requests or static microbatch; no paged continuous batching"},
        "constraints": ["human-supplied initial engine", "Linux Landlock x86_64/aarch64",
                        "GGUF/manifest/prequantized checkpoints need an explicit Python adapter",
                        "precision search needs an allowlist and frozen independent quality evaluation",
                        "declared capabilities do not certify every model/device/library combination"],
    }


def normalize_engine(config, project):
    engine = config.setdefault("engine", {})
    if not isinstance(engine, dict):
        raise ValueError("engine must be an object")
    source = engine.get("initial_source", config.get("engine_template", "templates/engine/transformers"))
    source = (Path(project) / source).resolve()
    if not (source / "engine.py").is_file() or not (source / "engine.json").is_file():
        raise ValueError(f"Initial engine must supply engine.py and engine.json: {source}")
    engine["initial_source"] = str(source)
    engine["initial_source_hash"] = source_hash(source)
    config["engine_template"] = str(source)
    engine.setdefault("adapter", "python" if config["model"].get("fixture") else "transformers")
    if engine["adapter"] not in ("python", "transformers"):
        raise ValueError("engine.adapter must be python or transformers; wrap other runtimes in engine.py")
    engine.setdefault("model_format", "safetensors")
    if engine["model_format"] not in ("safetensors", "gguf", "manifest"):
        raise ValueError("engine.model_format must be safetensors, gguf or manifest")
    engine.setdefault("model_loader", "auto")
    if engine["model_loader"] not in ("auto", "causal_lm", "image_text"):
        raise ValueError("Unknown engine.model_loader")
    if engine["adapter"] == "transformers" and engine["model_format"] != "safetensors":
        raise ValueError("GGUF/manifest models require an explicitly supplied python adapter")
    engine.setdefault("supported_devices", ["cpu", "cuda", "rocm"])
    engine.setdefault("supported_quantization", sorted(QUANTIZATION))
    engine.setdefault("required_packages", ["torch", "transformers", "safetensors"]
                      if engine["adapter"] == "transformers" else [])
    for key in ("supported_devices", "supported_quantization", "required_packages"):
        if not isinstance(engine[key], list) or not all(isinstance(v, str) and v for v in engine[key]):
            raise ValueError(f"engine.{key} must be a list of names")
    command = engine.setdefault("build_command", [])
    if not isinstance(command, list) or not all(isinstance(v, str) and v for v in command):
        raise ValueError("engine.build_command must be an argv list (no shell)")
    timeout = engine.setdefault("build_timeout_seconds", 300)
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600:
        raise ValueError("engine.build_timeout_seconds must be in (0, 3600]")
    return engine


def compatibility_errors(config, precision):
    engine = config["engine"]
    device = config["model"].get("device", config["hardware"].get("device", "cpu")).split(":", 1)[0]
    errors = []
    if device not in engine["supported_devices"]:
        errors.append(f"adapter_device_unsupported:{device}")
    if precision["quantization"] not in engine["supported_quantization"]:
        errors.append(f"adapter_quantization_unsupported:{precision['quantization']}")
    if engine["adapter"] == "transformers":
        if precision["kv_cache_dtype"] != "auto" or precision["accumulation_dtype"] != "auto":
            errors.append("explicit_cache_or_accumulation_precision_requires_custom_adapter")
        if precision["quantization"].startswith("bnb-") and device != "cuda":
            errors.append("builtin_bitsandbytes_adapter_requires_cuda")
    return errors
