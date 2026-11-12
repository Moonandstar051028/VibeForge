"""Canonical runtime precision, independent of checkpoint storage format."""

from copy import deepcopy


FLOAT_DTYPES = {"float32": 4, "float16": 2, "bfloat16": 2}
ALIASES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
QUANTIZATION = {"none", "bnb-nf4", "bnb-fp4", "bnb-int8"}


def normalize_precision(value=None, model=None):
    model = model or {}
    value = deepcopy({} if value is None else value)
    if not isinstance(value, dict):
        raise ValueError("precision must be an object or a JSON profile reference")
    dtype = value.get("compute_dtype", model.get("dtype", "bfloat16"))
    dtype = ALIASES.get(dtype, dtype)
    if dtype not in FLOAT_DTYPES:
        raise ValueError(f"Unsupported compute dtype: {dtype}")
    quantization = value.get("quantization", model.get("quantization")) or "none"
    quantization = {"bitsandbytes-4bit": "bnb-nf4", "bitsandbytes-8bit": "bnb-int8"}.get(quantization, quantization)
    if not isinstance(quantization, str) or not (
        quantization in QUANTIZATION or quantization.startswith("custom:") and len(quantization) > 7
    ):
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    weight_dtype = "int4" if quantization in ("bnb-nf4", "bnb-fp4") else (
        "int8" if quantization == "bnb-int8" else dtype)
    weight_dtype = value.get("weight_dtype", weight_dtype)
    if not isinstance(weight_dtype, str) or not weight_dtype:
        raise ValueError("precision.weight_dtype must be a nonempty name")
    if quantization == "none" and weight_dtype != dtype:
        raise ValueError("Unquantized weight_dtype must match compute_dtype")
    if quantization in ("bnb-nf4", "bnb-fp4", "bnb-int8") and weight_dtype != (
        "int8" if quantization == "bnb-int8" else "int4"
    ):
        raise ValueError("Quantization and weight_dtype disagree")
    result = {
        "weight_dtype": weight_dtype, "compute_dtype": dtype,
        "quantization": quantization,
        "kv_cache_dtype": value.get("kv_cache_dtype", "auto"),
        "accumulation_dtype": value.get("accumulation_dtype", "auto"),
        "double_quant": value.get("double_quant", True),
    }
    for key in ("kv_cache_dtype", "accumulation_dtype"):
        if result[key] not in {"auto", *FLOAT_DTYPES, "int8", "fp8"}:
            raise ValueError(f"Invalid precision.{key}")
    if not isinstance(result["double_quant"], bool):
        raise ValueError("precision.double_quant must be boolean")
    unknown = set(value) - set(result)
    if unknown:
        raise ValueError(f"Unknown precision fields: {sorted(unknown)}")
    return result


def engine_precision(config):
    return normalize_precision(config.get("precision"), config)


def precision_fields(precision):
    """Keep legacy engine adapters usable while recording the full contract."""
    return {"precision": precision, "dtype": precision["compute_dtype"],
            "quantization": None if precision["quantization"] == "none" else precision["quantization"]}


def validate_candidate_runtime(parent, candidate, campaign):
    for key in ("backend", "model_path", "device", "max_context", "model_loader", "thinking"):
        if parent.get(key) != candidate.get(key):
            raise ValueError(f"Candidate changed fixed engine field: {key}")
    precision = engine_precision(candidate)
    # Conflicting legacy aliases must not make the recorded and loaded precision differ.
    if "dtype" in candidate and candidate["dtype"] != precision["compute_dtype"]:
        raise ValueError("Candidate dtype disagrees with precision.compute_dtype")
    legacy = candidate.get("quantization") or "none"
    legacy = {"bitsandbytes-4bit": "bnb-nf4", "bitsandbytes-8bit": "bnb-int8"}.get(legacy, legacy)
    if legacy != precision["quantization"]:
        raise ValueError("Candidate quantization disagrees with precision contract")
    previous = engine_precision(parent)
    scheduler = candidate.get("request_scheduler", "serialized_lock")
    if scheduler not in {"serialized_lock", "concurrent_requests", "microbatch"}:
        raise ValueError("Candidate request_scheduler must be serialized_lock, concurrent_requests or microbatch")
    if (scheduler != parent.get("request_scheduler", "serialized_lock")
            and "scheduler" not in campaign.get("optimization", {}).get("allowed_surfaces", [])):
        raise ValueError("Candidate scheduler change was not authorized by this campaign")
    if precision != previous:
        policy = campaign.get("optimization", {})
        if not policy.get("allow_precision_changes"):
            raise ValueError("Candidate precision change was not authorized by this campaign")
        if precision not in policy.get("allowed_precisions", []):
            raise ValueError("Candidate precision is outside the frozen allowlist")
    return precision
