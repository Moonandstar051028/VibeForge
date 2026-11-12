import importlib.util
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

class Batch(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return self


class Vector:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)

    def bool(self):
        return [bool(value) for value in self.values]

    def __getitem__(self, key):
        if isinstance(key, list):
            return Vector(value for value, keep in zip(self.values, key) if keep)
        return self.values[key]


class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    @property
    def shape(self):
        return len(self.rows), len(self.rows[0])

    def sum(self, dim=None):
        assert dim == 1
        return Vector(sum(row) for row in self.rows)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, column = key
            return Vector(self.rows[row][column])
        return Vector(self.rows[key])


class InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Torch:
    @staticmethod
    def inference_mode():
        return InferenceMode()


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, prompts, return_tensors="pt", padding=False):
        rows = [[ord(character) % 31 + 1 for character in prompt] for prompt in (
            prompts if isinstance(prompts, list) else [prompts])]
        width = max(len(row) for row in rows)
        padded = [[0] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return Batch(input_ids=Matrix(padded), attention_mask=Matrix(masks))

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


class Model:
    device = "cpu"
    generation_config = SimpleNamespace(eos_token_id=0)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
        return Matrix([row + [7] * max_new_tokens for row in input_ids.rows])


def load_template(project):
    path = Path(project) / "templates/engine/transformers/engine.py"
    spec = importlib.util.spec_from_file_location("microbatch_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONFIG = {"request_scheduler": "microbatch", "microbatch_max_size": 2,
                     "microbatch_wait_ms": 50, "max_new_tokens": 8, "max_context": 64,
                     "device": "cpu", "use_cache": True}
    module.MODEL, module.TOKENIZER, module.TORCH = Model(), Tokenizer(), Torch()
    module.BATCH_QUEUE = queue.Queue()
    threading.Thread(target=module._microbatch_worker, daemon=True).start()
    return module


def test_microbatch_worker_batches_compatible_requests_and_preserves_inputs(project):
    engine = load_template(project)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda prompt: engine.generate(prompt, max_new_tokens=3, fixed_length=True),
            ["short", "a longer prompt"],
        ))

    assert [result["input_ids"] for result in results] == [
        [ord(character) % 31 + 1 for character in prompt]
        for prompt in ["short", "a longer prompt"]]
    assert all(result["token_ids"] == [7, 7, 7] for result in results)
    assert all(result["batch_size"] == 2 for result in results)
    assert all(result["batching_mode"] == "static_microbatch" for result in results)


def test_openai_zero_temperature_selects_greedy_generation(project):
    engine = load_template(project)

    greedy = engine._generation_options(3, False, {"temperature": 0, "top_p": 0.9})
    sampled = engine._generation_options(3, False, {"temperature": 0.6, "top_p": 0.9})

    assert greedy["do_sample"] is False
    assert "temperature" not in greedy and "top_p" not in greedy
    assert sampled["do_sample"] is True
    assert sampled["temperature"] == 0.6 and sampled["top_p"] == 0.9
    with pytest.raises(ValueError, match="non-negative"):
        engine._generation_options(3, False, {"temperature": -0.1})
