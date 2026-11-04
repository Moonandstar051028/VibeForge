"""Deterministic CPU fixture. Its timings never represent Qwen performance."""

import json
import time

CONFIG = {}


def load(config):
    global CONFIG
    CONFIG = config


def logits(input_ids, indices=None):
    work = sum(range(120000))
    seed = sum((index + 1) * token for index, token in enumerate(input_ids)) + work
    indices = list(range(32)) if indices is None else indices
    return {"indices": indices, "values": [((seed * (index + 3) + index * index) % 997) / 997 for index in indices]}


def generate(prompt, max_new_tokens=16, fixed_length=False):
    started = time.perf_counter()
    input_ids = [ord(character) % 128 for character in prompt] or [1]
    tokens, times = [], []
    for _ in range(max_new_tokens):
        scores = logits(input_ids + tokens)
        tokens.append(max(range(32), key=scores["values"].__getitem__))
        times.append(time.perf_counter())
    return {
        "text": " ".join(map(str, tokens)), "input_ids": input_ids, "token_ids": tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "ttft_seconds": times[0] - started if times else None,
        "tpot_seconds": (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None,
        "fixture": True,
    }


def chat(messages, max_new_tokens=128):
    text = json.dumps({"tool": "submit_candidate", "hypothesis": "toy protocol ready"})
    return {"text": text, "token_ids": list(range(min(len(text), max_new_tokens))), "fixture": True}
