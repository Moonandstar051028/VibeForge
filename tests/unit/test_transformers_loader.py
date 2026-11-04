import importlib.util
import sys
from types import SimpleNamespace

import pytest

from vibeforge.contracts.precision import normalize_precision, precision_fields


@pytest.mark.parametrize("quant", ["none", "bnb-nf4", "bnb-fp4", "bnb-int8"])
@pytest.mark.parametrize("architecture,image_text", [("LlamaForCausalLM", False),
                                                     ("Qwen3_5MoeForConditionalGeneration", True)])
def test_loader_selects_model_class_and_quantizer(project, monkeypatch, quant, architecture, image_text):
    calls = []
    class Model:
        def eval(self):
            return self
        def to(self, device):
            return self
    def loader(name):
        return SimpleNamespace(from_pretrained=lambda path, **kwargs: calls.append((name, kwargs)) or Model())
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(architectures=[architecture])),
        AutoModelForCausalLM=loader("causal"), AutoModelForImageTextToText=loader("image_text"),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        BitsAndBytesConfig=lambda **kwargs: kwargs,
    )
    torch = SimpleNamespace(float32="fp32", float16="fp16", bfloat16="bf16", set_num_threads=lambda n: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    spec = importlib.util.spec_from_file_location("initial_engine", project / "templates/engine/transformers/engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load({"model_path": "/local/model", "device": "cuda:0",
                 **precision_fields(normalize_precision({"quantization": quant}))})
    name, kwargs = calls[0]
    assert name == ("image_text" if image_text else "causal")
    assert kwargs["dtype"] == "bf16" and kwargs["device_map"] == "cuda:0"
    if quant == "none":
        assert "quantization_config" not in kwargs
    elif quant == "bnb-int8":
        assert kwargs["quantization_config"] == {"load_in_8bit": True}
    else:
        assert kwargs["quantization_config"]["bnb_4bit_quant_type"] == quant.removeprefix("bnb-")
