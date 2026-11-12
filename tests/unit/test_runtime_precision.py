from types import SimpleNamespace

import pytest

from vibeforge.contracts.precision import normalize_precision, precision_fields
from vibeforge.engine.backend import load_backend
from vibeforge.engine.precision import validate_loaded_quantization
from vibeforge.storage.atomic import write_json


def configuration(quantization="bnb-nf4", double_quant=True):
    return {"backend": "transformers", **precision_fields(normalize_precision({
        "quantization": quantization, "compute_dtype": "bfloat16", "double_quant": double_quant}))}


def quantized_layer(quant_type="nf4", nested=True, compute_dtype="torch.bfloat16"):
    return SimpleNamespace(weight=SimpleNamespace(quant_state=SimpleNamespace(
        quant_type=quant_type, nested=nested)), compute_dtype=compute_dtype)


def backend(*layers, loaded=True):
    return SimpleNamespace(MODEL=SimpleNamespace(is_loaded_in_4bit=loaded, modules=lambda: layers))


@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("double_quant", [False, True])
def test_loaded_weight_state_matches_the_declared_profile(quant_type, double_quant):
    result = validate_loaded_quantization(backend(
        SimpleNamespace(weight=None), quantized_layer(quant_type, double_quant),
        quantized_layer(quant_type, double_quant)), configuration("bnb-" + quant_type, double_quant))
    assert result["quantization"] == "bnb-" + quant_type
    assert result["double_quant"] is double_quant
    assert result["checked_layers"] == 2


@pytest.mark.parametrize("bad_layer,field", [
    (quantized_layer(nested=False), "double_quant"),
    (quantized_layer(quant_type="fp4"), "quantization type"),
    (quantized_layer(compute_dtype="torch.float16"), "compute_dtype"),
    (quantized_layer(compute_dtype=None), "compute_dtype"),
])
def test_one_inconsistent_layer_cannot_hide_behind_a_matching_layer(bad_layer, field):
    with pytest.raises(ValueError, match=field):
        validate_loaded_quantization(backend(quantized_layer(), bad_layer), configuration())


@pytest.mark.parametrize("model,reason", [
    (SimpleNamespace(), "no inspectable"),
    (backend(loaded=False), "not loaded in 4-bit"),
    (backend(SimpleNamespace(weight=None)), "no materialized"),
])
def test_missing_loaded_precision_evidence_is_rejected(model, reason):
    with pytest.raises(ValueError, match=reason):
        validate_loaded_quantization(model, configuration())


def test_unmaterialized_quantized_weight_is_rejected():
    weight = type("Params4bit", (), {"quant_state": None})()
    with pytest.raises(ValueError, match="unmaterialized"):
        validate_loaded_quantization(backend(SimpleNamespace(weight=weight)), configuration())


def test_non_four_bit_backends_are_outside_this_check():
    assert validate_loaded_quantization(SimpleNamespace(), configuration("none")) is None
    assert validate_loaded_quantization(SimpleNamespace(), {"backend": "toy"}) is None


def test_candidate_cannot_change_the_transport_precision_contract(tmp_path):
    write_json(tmp_path / "engine.json", configuration())
    (tmp_path / "engine.py").write_text('''
def load(config):
    config["precision"]["double_quant"] = False
def generate(**kwargs): pass
def chat(**kwargs): pass
def chat_completion(**kwargs): pass
def logits(**kwargs): pass
''')
    _, contract = load_backend(tmp_path)
    assert contract["precision"]["double_quant"] is True
