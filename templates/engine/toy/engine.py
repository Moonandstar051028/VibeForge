"""Deterministic CPU fixture. Its timings never represent Qwen performance."""

import json
import time

CONFIG = {}


class _ToyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(character) % 128 for character in text] or [1]}

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True, **kwargs):
        value = json.dumps({"messages": messages, "tools": tools or [],
                            "add_generation_prompt": add_generation_prompt}, ensure_ascii=False)
        return [ord(character) % 128 for character in value] if tokenize else value


TOKENIZER = _ToyTokenizer()


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


def chat_completion(messages, tools=None, max_tokens=None, **kwargs):
    functions = [tool.get("function", tool) for tool in (tools or [])]
    context = {}
    for message in messages:
        if message.get("role") == "user":
            try:
                context = json.loads(message.get("content", ""))
            except (ValueError, TypeError):
                pass
    if any("matching this schema" in str(message.get("content", "")) and '"ready"' in str(message.get("content", ""))
           for message in messages):
        return {"text": json.dumps({"ready": True, "message": "toy Codex transport ready"}),
                "tool_calls": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    if any(function.get("name") == "report_blocked" for function in functions):
        return {"text": "", "tool_calls": [{"name": "report_blocked", "arguments": json.dumps({
            "reason": "Toy transport fixture does not autonomously propose code changes."})}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    if any(function.get("name") == "submit_candidate" for function in functions):
        previous = context.get("previous")
        reflection = ({"iteration": previous["iteration"], "hypothesis_status": "inconclusive",
                       "explanation": "Toy transport fixture; performance reasoning is not tested.",
                       "next_action": "Use the configured integration fixture.",
                       "evidence_paths": [f"iterations/{previous['iteration']:04d}/decision.json"]}
                      if previous else None)
        return {
            "text": "",
            "tool_calls": [{
                "name": "submit_candidate",
                "arguments": json.dumps({
                    "hypothesis": "Toy backend verified the Qwen-Agent tool-call transport.",
                    "previous_analysis": "The deterministic fixture confirms the local function-call path; it does not evaluate Qwen reasoning.",
                    "summary": "Transport smoke test only.",
                    "reflection": reflection or {},
                }),
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    return {"text": "toy completion ready", "tool_calls": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
