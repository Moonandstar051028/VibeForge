"""Check loaded bitsandbytes state independently of candidate configuration claims."""

from vibeforge.contracts.precision import engine_precision


def validate_loaded_quantization(backend, config):
    """Attest the supported Transformers 4-bit loader before serving requests.

    Checking engine.json alone is insufficient: candidate Python can override the
    BitsAndBytesConfig arguments while leaving the recorded profile unchanged.
    Inspect materialized weights and layer compute types, not just model.config.
    This checks the NF4/FP4 weight contract, not KV-cache or accumulator precision.
    """
    expected = engine_precision(config)
    if config.get("backend") != "transformers" or expected["quantization"] not in {"bnb-nf4", "bnb-fp4"}:
        return None

    model = getattr(backend, "MODEL", None)
    if model is None or not callable(getattr(model, "modules", None)):
        raise ValueError("Loaded precision mismatch: no inspectable Transformers model")
    if not getattr(model, "is_loaded_in_4bit", False):
        raise ValueError("Loaded precision mismatch: model is not loaded in 4-bit")

    quant_type = expected["quantization"].removeprefix("bnb-")
    count = 0
    for layer in model.modules():
        weight = getattr(layer, "weight", None)
        state = getattr(weight, "quant_state", None)
        if state is None:
            # A Params4bit without materialized state must not escape inspection.
            if type(weight).__name__ == "Params4bit":
                raise ValueError("Loaded precision mismatch: unmaterialized 4-bit weight")
            continue
        count += 1
        if getattr(state, "quant_type", None) != quant_type:
            raise ValueError("Loaded precision mismatch: 4-bit quantization type disagrees with engine.json")
        if getattr(state, "nested", None) is not expected["double_quant"]:
            raise ValueError("Loaded precision mismatch: double_quant disagrees with engine.json")
        compute_dtype = str(getattr(layer, "compute_dtype", None)).removeprefix("torch.")
        if compute_dtype != expected["compute_dtype"]:
            raise ValueError("Loaded precision mismatch: 4-bit compute_dtype disagrees with engine.json")
    if not count:
        raise ValueError("Loaded precision mismatch: no materialized 4-bit weights found")
    return {"quantization": expected["quantization"], "compute_dtype": expected["compute_dtype"],
            "double_quant": expected["double_quant"], "checked_layers": count,
            "verified_fields": ["quantization", "compute_dtype", "double_quant"]}
