"""User-supplied Transformers bootstrap for local autoregressive text models."""

import time
import json
import re
import copy
import ast
import queue
import threading

CONFIG = {}
MODEL = TOKENIZER = TORCH = None
BATCH_QUEUE = None
MICROBATCH_SUPPORTED = True


def load(config):
    global CONFIG, MODEL, TOKENIZER, TORCH, BATCH_QUEUE
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
    if config.get("attn_implementation"):
        arguments["attn_implementation"] = config["attn_implementation"]
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
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    TOKENIZER.padding_side = "left"
    if config.get("request_scheduler") == "microbatch":
        BATCH_QUEUE = queue.Queue()
        threading.Thread(target=_microbatch_worker, name="vibeforge-microbatch", daemon=True).start()


def synchronize():
    if CONFIG.get("device", "cpu").startswith(("cuda", "rocm")):
        TORCH.cuda.synchronize()


def _generation_finish_reason(tokens, max_new_tokens):
    eos = getattr(getattr(MODEL, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = TOKENIZER.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple, set)) else [eos])
    return "stop" if tokens and tokens[-1] in eos else (
        "length" if len(tokens) >= max_new_tokens else "stop")


def _generation_options(max_new_tokens, fixed_length, generation_options):
    if not 1 <= max_new_tokens <= CONFIG.get("max_new_tokens", 1024):
        raise ValueError("Output length exceeds configured limit")
    options = {"max_new_tokens": max_new_tokens, "do_sample": False,
               "use_cache": CONFIG.get("use_cache", True),
               "pad_token_id": TOKENIZER.eos_token_id}
    generation_options = generation_options or {}
    unknown = set(generation_options) - {"temperature", "top_p", "top_k", "min_p"}
    if unknown:
        raise ValueError(f"Unsupported generation options: {sorted(unknown)}")
    if generation_options:
        temperature = generation_options.get("temperature", 1.0)
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or temperature < 0):
            raise ValueError("Generation temperature must be a non-negative number")
        # OpenAI-compatible clients use temperature=0 to request deterministic
        # greedy decoding. Sampling controls are meaningful only above zero.
        if temperature > 0:
            options.update(do_sample=True, temperature=float(temperature))
            for name in ("top_p", "top_k", "min_p"):
                if name in generation_options:
                    options[name] = generation_options[name]
    if CONFIG.get("cache_implementation"):
        options["cache_implementation"] = CONFIG["cache_implementation"]
    if fixed_length:
        options["min_new_tokens"] = max_new_tokens
    return options


def _generate_direct(prompt, max_new_tokens=16, fixed_length=False, skip_special_tokens=True,
                     generation_options=None):
    started = time.perf_counter()
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    if inputs.input_ids.shape[-1] + max_new_tokens > CONFIG.get("max_context", 4096):
        raise ValueError("Request exceeds configured context")
    options = _generation_options(max_new_tokens, fixed_length, generation_options)
    synchronize()
    with TORCH.inference_mode():
        result = MODEL.generate(**inputs, **options)
    synchronize()
    tokens = result[0, inputs.input_ids.shape[-1]:].tolist()
    return {
        "text": TOKENIZER.decode(tokens, skip_special_tokens=skip_special_tokens),
        "input_ids": inputs.input_ids[0].tolist(), "token_ids": tokens,
        "finish_reason": _generation_finish_reason(tokens, max_new_tokens),
        "elapsed_seconds": time.perf_counter() - started,
        "ttft_seconds": None,
        "tpot_seconds": None,
        "batch_size": 1,
        "batching_mode": "none",
    }


def _generate_batch(requests):
    started = time.perf_counter()
    first = requests[0]["arguments"]
    prompts = [request["arguments"]["prompt"] for request in requests]
    inputs = TOKENIZER(prompts, return_tensors="pt", padding=True).to(MODEL.device)
    input_lengths = inputs.attention_mask.sum(dim=1).tolist()
    if max(input_lengths) + first["max_new_tokens"] > CONFIG.get("max_context", 4096):
        raise ValueError("Request exceeds configured context")
    options = _generation_options(
        first["max_new_tokens"], first["fixed_length"], first["generation_options"])
    synchronize()
    with TORCH.inference_mode():
        generated = MODEL.generate(**inputs, **options)
    synchronize()
    elapsed = time.perf_counter() - started
    prompt_width = inputs.input_ids.shape[-1]
    results = []
    for index, request in enumerate(requests):
        input_ids = inputs.input_ids[index][inputs.attention_mask[index].bool()].tolist()
        tokens = generated[index, prompt_width:].tolist()
        results.append({
            "text": TOKENIZER.decode(tokens, skip_special_tokens=request["arguments"]["skip_special_tokens"]),
            "input_ids": input_ids, "token_ids": tokens,
            "finish_reason": _generation_finish_reason(tokens, first["max_new_tokens"]),
            "elapsed_seconds": elapsed, "ttft_seconds": None, "tpot_seconds": None,
            "batch_size": len(requests), "batching_mode": "static_microbatch",
            "batch_queue_seconds": started - request["enqueued_at"],
        })
    return results


def _batch_signature(arguments):
    return json.dumps({key: arguments[key] for key in (
        "max_new_tokens", "fixed_length", "skip_special_tokens", "generation_options")},
        sort_keys=True, separators=(",", ":"))


def _microbatch_worker():
    pending = []
    maximum = max(1, int(CONFIG.get("microbatch_max_size", 4)))
    wait_seconds = max(0.0, float(CONFIG.get("microbatch_wait_ms", 4))) / 1000
    while True:
        first = pending.pop(0) if pending else BATCH_QUEUE.get()
        batch = [first]
        deadline = time.perf_counter() + wait_seconds
        while len(batch) < maximum:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                candidate = BATCH_QUEUE.get(timeout=remaining)
            except queue.Empty:
                break
            if candidate["signature"] == first["signature"]:
                batch.append(candidate)
            else:
                pending.append(candidate)
        try:
            results = _generate_batch(batch)
            for request, result in zip(batch, results):
                request["result"] = result
        except Exception as error:
            for request in batch:
                request["error"] = error
        finally:
            for request in batch:
                request["event"].set()


def generate(prompt, max_new_tokens=16, fixed_length=False, skip_special_tokens=True,
             generation_options=None):
    if CONFIG.get("request_scheduler") != "microbatch":
        return _generate_direct(prompt, max_new_tokens, fixed_length, skip_special_tokens, generation_options)
    arguments = {
        "prompt": prompt, "max_new_tokens": max_new_tokens, "fixed_length": fixed_length,
        "skip_special_tokens": skip_special_tokens, "generation_options": generation_options or {},
    }
    request = {"arguments": arguments, "signature": _batch_signature(arguments),
               "enqueued_at": time.perf_counter(), "event": threading.Event()}
    BATCH_QUEUE.put(request)
    timeout = float(CONFIG.get("microbatch_request_timeout_seconds", 600))
    if not request["event"].wait(timeout):
        raise TimeoutError("Microbatch generation timed out")
    if request.get("error") is not None:
        raise request["error"]
    return request["result"]


def _thinking_settings(requested=None):
    configured = CONFIG.get("thinking") or {}
    enabled = configured.get("enabled", False) if requested is None else requested
    if not isinstance(enabled, bool):
        raise ValueError("chat_template_kwargs.enable_thinking must be boolean")
    if not enabled:
        return False, {}
    return True, {
        "temperature": configured.get("temperature", 0.6),
        "top_p": configured.get("top_p", 0.95),
        "top_k": configured.get("top_k", 20),
        "min_p": configured.get("min_p", 0.0),
    }


def chat(messages, max_new_tokens=512):
    prompt = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return generate(prompt, max_new_tokens)


def _find_sequence(values, marker):
    if not marker:
        return None
    for index in range(len(values) - len(marker) + 1):
        if values[index:index + len(marker)] == marker:
            return index
    return None


def split_reasoning(text, token_ids, enabled):
    """Separate Qwen's explicit thinking segment from user-visible content."""
    marker = []
    if hasattr(TOKENIZER, "encode"):
        marker = TOKENIZER.encode("</think>", add_special_tokens=False)
    marker_index = _find_sequence(token_ids, marker)
    if marker_index is not None:
        reasoning_tokens = marker_index + len(marker)
        visible_tokens = len(token_ids) - reasoning_tokens
    elif enabled:
        reasoning_tokens, visible_tokens = len(token_ids), 0
    else:
        reasoning_tokens, visible_tokens = 0, len(token_ids)

    if "</think>" in text:
        reasoning, visible = text.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
        return reasoning, visible, reasoning_tokens, visible_tokens, True
    if enabled:
        return text.removeprefix("<think>").strip(), "", reasoning_tokens, visible_tokens, False
    return "", text, reasoning_tokens, visible_tokens, True


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
                try:
                    value = json.loads(value)
                except (ValueError, TypeError):
                    # Local instruction models sometimes emit Python literals
                    # (False, single-quoted dicts) inside Qwen's XML envelope.
                    value = ast.literal_eval(value)
                expected = schema.get("type")
                valid = {
                    "boolean": lambda item: isinstance(item, bool),
                    "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
                    "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
                    "object": lambda item: isinstance(item, dict),
                    "array": lambda item: isinstance(item, list),
                }.get(expected, lambda item: True)(value)
                if not valid:
                    raise ValueError(f"Tool parameter {name} must be {expected}")
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
    tool_choice = kwargs.get("tool_choice")
    required_tool = None
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise ValueError("Only named function tool_choice is supported")
        required_tool = tool_choice.get("function", {}).get("name")
    elif isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
        required_tool = tool_choice
    if required_tool:
        tools = [tool for tool in (tools or [])
                 if tool.get("function", {}).get("name") == required_tool]
        if not tools:
            raise ValueError(f"Required tool is unavailable: {required_tool}")
        instruction = f"You must call the {required_tool} function now. Do not answer in prose."
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0].get('content', '')}\n\n{instruction}"
        else:
            messages.insert(0, {"role": "system", "content": instruction})
    elif tool_choice == "required":
        instruction = "You must call one of the available functions now. Do not answer in prose."
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0].get('content', '')}\n\n{instruction}"
        else:
            messages.insert(0, {"role": "system", "content": instruction})
    elif tool_choice == "none":
        tools = None
    for message in messages:
        for call in message.get("tool_calls", []):
            arguments = call["function"].get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be JSON objects")
            call["function"]["arguments"] = arguments
    template_options = kwargs.get("chat_template_kwargs") or {}
    if not isinstance(template_options, dict):
        raise ValueError("chat_template_kwargs must be an object")
    unknown_template_options = set(template_options) - {"enable_thinking"}
    if unknown_template_options:
        raise ValueError(f"Unsupported chat template options: {sorted(unknown_template_options)}")
    enabled, sampling = _thinking_settings(template_options.get("enable_thinking"))
    for name in ("temperature", "top_p", "top_k", "min_p"):
        if kwargs.get(name) is not None:
            sampling[name] = kwargs[name]
    output_limit = max_tokens if max_tokens is not None else kwargs.get("max_completion_tokens")

    def generate_reply(thinking_enabled, generation_options):
        prompt = TOKENIZER.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_enabled,
        )
        generated = generate(
            prompt,
            max_new_tokens=output_limit if output_limit is not None else CONFIG.get("max_new_tokens", 512),
            skip_special_tokens=False,
            generation_options=generation_options,
        )
        parsed = split_reasoning(generated["text"], generated["token_ids"], thinking_enabled)
        return generated, parsed

    generated, parsed = generate_reply(enabled, sampling)
    reasoning, text, reasoning_tokens, visible_tokens, thinking_complete = parsed
    thinking_recovery = False
    if enabled and not thinking_complete:
        # A malformed/incomplete native thinking segment must not poison the
        # surrounding Responses tool loop. Retry once with a plain assistant prefix.
        generated, parsed = generate_reply(False, {})
        reasoning, text, reasoning_tokens, visible_tokens, thinking_complete = parsed
        thinking_recovery = True
    text = text.strip()
    text = text.removesuffix("<|im_end|>").removesuffix("<|endoftext|>").strip()
    calls = parse_tool_calls(text, tools)
    if "<tool_call>" in text and not calls:
        raise ValueError("Malformed or truncated Qwen tool call")
    if calls:
        text = text.split("<tool_call>", 1)[0].strip()
    finish_reason = generated.get("finish_reason", "stop")
    if thinking_recovery and not (text or calls):
        finish_reason = "incomplete_thinking"
    elif enabled and not thinking_complete:
        finish_reason = "incomplete_thinking"
    return {
        "text": text,
        "reasoning_content": reasoning or None,
        "tool_calls": [{"name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False)} for call in calls],
        "finish_reason": finish_reason,
        "thinking": {"enabled": enabled, "complete": thinking_complete,
                     "recovered_without_thinking": thinking_recovery,
                     "reasoning_tokens": reasoning_tokens, "visible_tokens": visible_tokens},
        "usage": {
            "prompt_tokens": len(generated["input_ids"]),
            "completion_tokens": len(generated["token_ids"]),
            "total_tokens": len(generated["input_ids"]) + len(generated["token_ids"]),
            "completion_tokens_details": {
                "reasoning_tokens": reasoning_tokens,
                "visible_tokens": visible_tokens,
            },
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
