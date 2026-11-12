"""Temporary host instrumentation of the ordinary generate path.

CUDA events are resolved at the existing request boundary, never by synchronizing
each token. Internal token readiness is distinct from HTTP streaming latency.
"""

import time


def _tensor_bytes(value, seen=None):
    """Count logical tensor payload bytes in legacy or Cache-style KV structures."""
    seen = seen if seen is not None else set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if callable(getattr(value, "numel", None)) and callable(getattr(value, "element_size", None)):
        try:
            return int(value.numel()) * int(value.element_size())
        except (TypeError, ValueError, RuntimeError):
            return 0
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, seen) for item in value)
    cache_values = []
    for name in ("key_cache", "value_cache"):
        item = getattr(value, name, None)
        if item is not None:
            cache_values.append(item)
    return sum(_tensor_bytes(item, seen) for item in cache_values)


class _TokenizerProxy:
    def __init__(self, tokenizer, durations):
        self.tokenizer, self.durations = tokenizer, durations

    def _call(self, name, *args, **kwargs):
        started = time.perf_counter()
        try:
            method = self.tokenizer if name == "tokenize" else getattr(self.tokenizer, name)
            return method(*args, **kwargs)
        finally:
            self.durations[name] = self.durations.get(name, 0) + time.perf_counter() - started

    def __call__(self, *args, **kwargs):
        return self._call("tokenize", *args, **kwargs)

    def __getattr__(self, name):
        if name in {"decode", "batch_decode", "apply_chat_template"}:
            return lambda *args, **kwargs: self._call(name, *args, **kwargs)
        return getattr(self.tokenizer, name)


class RequestTelemetry:
    def __init__(self, backend, config, torch):
        self.backend, self.config, self.torch = backend, config, torch
        self.model = getattr(backend, "MODEL", None)
        self.cuda = torch is not None and str(config.get("device", "cpu")).startswith(("cuda", "rocm"))
        self.device = str(config.get("device", "cpu")).replace("rocm", "cuda", 1)
        self.handles, self.tokens, self.forwards, self.stack = [], [], [], []
        self.durations, self.generate_calls, self.cache_class = {}, 0, None
        self.kv_cache_bytes = None
        self.generate_kwargs = []
        self.tokenizer = None

    def stamp(self):
        if self.cuda:
            event = self.torch.cuda.Event(enable_timing=True)
            event.record(self.torch.cuda.current_stream(self.device))
            return event
        return time.perf_counter()

    def seconds(self, first, last):
        return first.elapsed_time(last) / 1000 if self.cuda else last - first

    def __enter__(self):
        try:
            return self._enter()
        except Exception:
            self.__exit__()
            raise

    def _enter(self):
        self.started = time.perf_counter()
        self.anchor = self.stamp()
        if self.cuda:
            self.torch.cuda.reset_peak_memory_stats(self.device)
        self.tokenizer = getattr(self.backend, "TOKENIZER", None)
        if self.tokenizer is not None:
            self.backend.TOKENIZER = _TokenizerProxy(self.tokenizer, self.durations)
        if self.model is not None and hasattr(self.model, "register_forward_pre_hook"):
            def before(module, args):
                self.stack.append((time.perf_counter(), self.stamp()))

            def after(module, args, output):
                if self.stack:
                    cpu, device = self.stack.pop()
                    self.forwards.append({"cpu_submission_seconds": time.perf_counter() - cpu,
                                          "start": device, "end": self.stamp()})
                cache = getattr(output, "past_key_values", None)
                if cache is not None:
                    self.cache_class = type(cache).__name__
                    self.kv_cache_bytes = _tensor_bytes(cache)

            self.handles.append(self.model.register_forward_pre_hook(before))
            self.handles.append(self.model.register_forward_hook(after, always_call=True))
            self.original_generate = self.model.generate
            self.had_override = "generate" in self.model.__dict__
            self.previous_override = self.model.__dict__.get("generate")
            from transformers import StoppingCriteria, StoppingCriteriaList
            owner = self

            class TokenReady(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    owner.tokens.append(owner.stamp())
                    return False

            def generate(*args, **kwargs):
                owner.generate_calls += 1
                owner.generate_kwargs.append({key: kwargs[key] for key in (
                    "do_sample", "use_cache", "cache_implementation", "max_new_tokens", "min_new_tokens",
                    "num_beams", "temperature", "top_p", "top_k")
                    if key in kwargs and isinstance(kwargs[key], (str, int, float, bool, type(None)))})
                kwargs["stopping_criteria"] = StoppingCriteriaList([*(kwargs.get("stopping_criteria") or []), TokenReady()])
                return owner.original_generate(*args, **kwargs)

            self.model.generate = generate
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        if hasattr(self, "original_generate"):
            if self.had_override:
                self.model.generate = self.previous_override
            else:
                del self.model.generate
        if self.tokenizer is not None:
            self.backend.TOKENIZER = self.tokenizer

    def report(self, output_tokens):
        trustworthy = self.generate_calls == 1 and len(self.tokens) == output_tokens and bool(self.tokens)
        gaps = [self.seconds(a, b) for a, b in zip(self.tokens, self.tokens[1:])] if trustworthy else []
        prefill = self.forwards[:1]
        decode = self.forwards[1:]
        memory = {"peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
                  "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device)} if self.cuda else None
        return {
            "ttft_seconds": self.seconds(self.anchor, self.tokens[0]) if trustworthy else None,
            "tpot_seconds": sum(gaps) / len(gaps) if gaps else None,
            "inter_token_seconds": gaps, "client_ttft_seconds": None,
            "token_timing_scope": "internal generation token-ready callback; CUDA stream events or CPU monotonic clock; not HTTP first byte",
            "token_timing_available": trustworthy, "observed_token_callbacks": len(self.tokens),
            "token_timing_unavailable_reason": None if trustworthy else "No single instrumentable MODEL.generate call with one callback per output token",
            "phases": {"tokenization_seconds": self.durations.get("tokenize"),
                       "detokenization_seconds": sum(self.durations.get(key, 0) for key in ("decode", "batch_decode")) if self.tokenizer is not None else None,
                       "chat_template_seconds": self.durations.get("apply_chat_template"),
                       "prefill_forward_seconds": sum(self.seconds(row["start"], row["end"]) for row in prefill) if prefill else None,
                       "decode_forward_seconds": sum(self.seconds(row["start"], row["end"]) for row in decode) if decode else None,
                       "forward_calls": len(self.forwards), "decode_steps": len(decode)},
            "phase_clock": "CUDA stream events including stream idle gaps" if self.cuda else "CPU perf_counter",
            "cuda_memory": memory, "observed_cache_class": self.cache_class,
            "kv_cache": {"logical_tensor_bytes": self.kv_cache_bytes,
                         "scope": "past_key_values tensors returned by the latest observed root forward; allocator overhead excluded"}
                        if self.cache_class is not None else None,
            "observed_generate_kwargs": self.generate_kwargs,
            "generate_kwargs_scope": "Scalar keyword arguments intercepted at MODEL.generate; omitted fields may inherit model defaults, and custom backends may be unobservable.",
            "instrumentation": "host tokenizer proxy, root forward hooks and token callback; enabled identically for both formal variants; no per-token synchronize",
            "unavailable_metrics": ["client streaming TTFT", "true DRAM bandwidth", "SM occupancy"]}
