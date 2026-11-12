import sys

import pytest

from vibeforge.contracts.precision import normalize_precision, precision_fields
from vibeforge.engine.process import EngineProcess
from vibeforge.storage.atomic import write_json


@pytest.mark.parametrize("actual_double_quant", [True, False])
def test_server_checks_actual_layer_state_before_becoming_ready(paths, tmp_path, actual_double_quant):
    """Transport fixture only: no real model or GPU is used in this test."""
    directory = tmp_path / "candidate"
    directory.mkdir()
    write_json(directory / "engine.json", {"backend": "transformers", "device": "cpu",
        **precision_fields(normalize_precision({"quantization": "bnb-nf4"}))})
    (directory / "engine.py").write_text('''
from types import SimpleNamespace
MODEL = None
def load(config):
    global MODEL
    config["precision"]["double_quant"] = ACTUAL
    layer = SimpleNamespace(weight=SimpleNamespace(quant_state=SimpleNamespace(
        quant_type="nf4", nested=ACTUAL)), compute_dtype="torch.bfloat16")
    MODEL = SimpleNamespace(is_loaded_in_4bit=True, modules=lambda: [layer])
def generate(**kwargs): pass
def chat(**kwargs): pass
def chat_completion(**kwargs): pass
def logits(**kwargs): pass
'''.replace("ACTUAL", repr(actual_double_quant)))
    engine = EngineProcess(paths, "runtime-precision", sys.executable, startup_timeout=10)
    log = tmp_path / "engine.log"
    try:
        if actual_double_quant:
            client = engine.start(directory, log)
            health = client.health()
            assert health["runtime_precision"]["double_quant"] is True
            assert health["runtime_precision"]["checked_layers"] == 1
        else:
            with pytest.raises(RuntimeError, match="Engine exited"):
                engine.start(directory, log)
            assert "Loaded precision mismatch: double_quant" in log.read_text()
    finally:
        engine.stop()
    assert not engine.record.exists()
