"""User-supplied Transformers bootstrap for local autoregressive text models."""

import time
import json
import re
import copy

CONFIG = {}
MODEL = TOKENIZER = TORCH = None


def load(config):
    global CONFIG, MODEL, TOKENIZER, TORCH
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    from vibeforge.contracts.precision import engine_precision
    CONFIG, TORCH = config, torch
    torch.set_num_threads(config.get("num_threads", 8))
    precision = engine_precision(config)
    dtype = getattr(torch, precision["compute_dtype"])
    device = config.get("device", "cpu").replace("rocm", "cuda", 1)
    arguments = {"local_files_only": True, "trust_remote_code": False,
                 "dtype": dtype}
    if device != "cpu" or precision["quantization"] != "none":
        arguments["device_map"] = device
    quantization = precision["quantization"]
    if quantization in ("bnb-nf4", "bnb-fp4", "bnb-int8"):
        from transformers import BitsAndBytesConfig
        arguments["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if quantization == "bnb-int8" else
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4" if quantization == "bnb-fp4" else "nf4",
                              bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=precision["double_quant"]))
    elif quantization != "none":
        raise ValueError("This Transformers template needs an explicit loader for custom quantization")
    if precision["kv_cache_dtype"] != "auto" or precision["accumulation_dtype"] != "auto":
        raise ValueError("Explicit KV/accumulation precision requires a dedicated backend")
    model_config = AutoConfig.from_pretrained(config["model_path"], local_files_only=True, trust_remote_code=False)
    if getattr(model_config, "quantization_config", None):
        raise ValueError("Prequantized checkpoints require an explicit adapter matching their stored quantization")
    loader = config.get("model_loader", "auto")
    if loader == "auto":
        loader = "image_text" if any("ForConditionalGeneration" in name for name in model_config.architectures or []) else "causal_lm"
    model_class = AutoModelForImageTextToText if loader == "image_text" else AutoModelForCausalLM
    MODEL = model_class.from_pretrained(config["model_path"], **arguments).eval()
    if "device_map" not in arguments:
        MODEL = MODEL.to(device)
    TOKENIZER = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True, trust_remote_code=False)


def synchronize():
    if CONFIG.get("device", "cpu").startswith(("cuda", "rocm")):
        TORCH.cuda.synchronize()


def generate(prompt, max_new_tokens=16, fixed_length=False, skip_special_tokens=True):
    if not 1 <= max_new_tokens <= CONFIG.get("max_new_tokens", 1024):
        raise ValueError("Output length exceeds configured limit")
    started = time.perf_counter()
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    if inputs.input_ids.shape[-1] + max_new_tokens > CONFIG.get("max_context", 4096):
        raise ValueError("Request exceeds configured context")
    synchronize()
    timestamps = []

    class TokenClock:
        first = True

        def put(self, value):
            if self.first:
                self.first = False
                return
            synchronize()
            timestamps.append(time.perf_counter())

        def end(self):
            pass

    options = {"max_new_tokens": max_new_tokens, "do_sample": False,
               "use_cache": CONFIG.get("use_cache", True), "streamer": TokenClock(),
               "pad_token_id": TOKENIZER.eos_token_id}
    if fixed_length:
        options["min_new_tokens"] = max_new_tokens
    with TORCH.inference_mode():
        result = MODEL.generate(**inputs, **options)
    synchronize()
    tokens = result[0, inputs.input_ids.shape[-1]:].tolist()
    return {
        "text": TOKENIZER.decode(tokens, skip_special_tokens=skip_special_tokens),
        "input_ids": inputs.input_ids[0].tolist(), "token_ids": tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "ttft_seconds": timestamps[0] - started if timestamps else None,
        "tpot_seconds": (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1) if len(timestamps) > 1 else None,
    }


def chat(messages, max_new_tokens=512):
    prompt = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return generate(prompt, max_new_tokens)


def parse_tool_calls(text, tools=None):
    schemas = {
        tool["function"]["name"]: tool["function"].get("parameters", {}).get("properties", {})
        for tool in (tools or [])
    }
    calls = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL):
        function = re.search(r"<function=([^>\n]+)>(.*?)</function>", block, flags=re.DOTALL)
        if not function:
            try:
                value = json.loads(block)
                calls.append({"name": value["name"], "arguments": value.get("arguments", {})})
            except (ValueError, KeyError, TypeError):
                continue
            continue
        arguments = {}
        function_name = function.group(1).strip()
        for name, raw_value in re.findall(
                r"<parameter=([^>\n]+)>(.*?)</parameter>", function.group(2), flags=re.DOTALL):
            name = name.strip()
            # The native template wraps each value with one newline. Preserve
            # meaningful indentation, blank lines, and numeric-looking strings.
            value = raw_value.removeprefix("\n").removesuffix("\n")
            schema = schemas.get(function_name, {}).get(name, {})
            if schema.get("type", "string") != "string":
                value = json.loads(value)
            arguments[name] = value
        calls.append({"name": function.group(1).strip(), "arguments": arguments})
    # Several non-Qwen chat templates emit an unwrapped JSON function call.
    if not calls:
        try:
            value = json.loads(text.strip())
            values = value.get("tool_calls", [value]) if isinstance(value, dict) else value
            for call in values:
                function = call.get("function", call)
                if function.get("name") in schemas:
                    arguments = function.get("arguments", function.get("parameters", {}))
                    calls.append({"name": function["name"], "arguments":
                                  json.loads(arguments) if isinstance(arguments, str) else arguments})
        except (ValueError, TypeError, AttributeError):
            pass
    return calls


def chat_completion(messages, tools=None, max_tokens=None, **kwargs):
    messages = copy.deepcopy(messages)
    for message in messages:
        for call in message.get("tool_calls", []):
            arguments = call["function"].get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be JSON objects")
            call["function"]["arguments"] = arguments
    prompt = TOKENIZER.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    generated = generate(
        prompt,
        max_new_tokens=max_tokens or kwargs.get("max_completion_tokens") or CONFIG.get("max_new_tokens", 512),
        skip_special_tokens=False,
    )
    text = re.sub(r"<think>.*?</think>", "", generated["text"], flags=re.DOTALL).strip()
    text = text.removesuffix("<|im_end|>").removesuffix("<|endoftext|>").strip()
    calls = parse_tool_calls(text, tools)
    if "<tool_call>" in text and not calls:
        raise ValueError("Malformed or truncated Qwen tool call")
    if calls:
        text = text.split("<tool_call>", 1)[0].strip()
    return {
        "text": text,
        "tool_calls": [{"name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False)} for call in calls],
        "usage": {
            "prompt_tokens": len(generated["input_ids"]),
            "completion_tokens": len(generated["token_ids"]),
            "total_tokens": len(generated["input_ids"]) + len(generated["token_ids"]),
        },
    }


def logits(input_ids, indices=None):
    if not input_ids or len(input_ids) > CONFIG.get("max_context", 4096):
        raise ValueError("Invalid logit input length")
    tokens = TORCH.tensor([input_ids], dtype=TORCH.long, device=MODEL.device)
    with TORCH.inference_mode():
        values = MODEL(input_ids=tokens, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    indices = list(range(min(32, values.shape[-1]))) if indices is None else indices
    return {"indices": indices, "values": values[indices].cpu().tolist()}
