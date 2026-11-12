"""Host-owned timing and bounded diagnostics, outside the editable engine source."""

import cProfile
import sys
import time
from vibeforge.engine.telemetry import RequestTelemetry
from vibeforge.engine.implementation import implementation_sources


def _torch(backend):
    return getattr(backend, "TORCH", None) or sys.modules.get("torch")


def _synchronize(backend, config):
    torch = _torch(backend)
    if torch is not None and str(config.get("device", "cpu")).startswith(("cuda", "rocm")):
        torch.cuda.synchronize(str(config.get("device", "cuda")).replace("rocm", "cuda", 1))


def _torch_summaries(events):
    """Keep CPU annotations, kernel events and CPU operator attribution separate."""
    rows = [{"name": event.key, "calls": event.count,
             "device_type": str(event.device_type).rsplit(".", 1)[-1],
             "input_shapes": getattr(event, "input_shapes", None),
             "self_cpu_seconds": event.self_cpu_time_total / 1e6,
             "self_device_seconds": getattr(event, "self_device_time_total", 0) / 1e6,
             "cpu_total_seconds": event.cpu_time_total / 1e6,
             "device_total_seconds": getattr(event, "device_time_total", 0) / 1e6}
            for event in events]
    cpu = [row for row in rows if row["device_type"] == "CPU"]
    phases = [row for row in cpu if row["name"].startswith("vibeforge.")]
    operators = [row for row in cpu if not row["name"].startswith("vibeforge.")]
    kernels = [row for row in rows if row["device_type"] != "CPU"
               and not row["name"].startswith("vibeforge.") and row["self_device_seconds"] > 0]
    return (phases, sorted(operators, key=lambda r: r["self_cpu_seconds"], reverse=True)[:15],
            sorted(kernels, key=lambda r: r["self_device_seconds"], reverse=True)[:15],
            sorted((r for r in operators if r["self_device_seconds"] > 0),
                   key=lambda r: r["self_device_seconds"], reverse=True)[:15])


def _tensor_signatures(values, limit=8):
    """Describe bounded module inputs without retaining tensors or profiler events."""
    result = []

    def visit(value):
        if len(result) >= limit:
            return
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                signature = {
                    "shape": [int(dimension) for dimension in shape],
                    "dtype": str(getattr(value, "dtype", "unknown")),
                    "device": str(getattr(value, "device", "unknown")),
                }
            except (TypeError, ValueError):
                signature = None
            if signature is not None and signature not in result:
                result.append(signature)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(values)
    return result


def observe_call(backend, config, operation, arguments, queue_seconds=0, detailed=True):
    """Same inference implementation, with host-owned clocks and read-only hooks."""
    started = time.perf_counter()
    concurrent = config.get("request_scheduler", "serialized_lock") in {"concurrent_requests", "microbatch"}
    if not concurrent:
        _synchronize(backend, config)
    if concurrent or not detailed:
        result = getattr(backend, operation)(**arguments)
        telemetry_report = {
            "ttft_seconds": None, "tpot_seconds": None, "inter_token_seconds": [],
            "client_ttft_seconds": None, "token_timing_available": False,
            "token_timing_unavailable_reason": (
                "Detailed hooks are disabled for concurrent requests because model/tokenizer monkeypatching is not thread-safe."),
            "phases": {}, "cuda_memory": None, "observed_cache_class": None,
            "observed_generate_kwargs": [], "kv_cache": None,
            "unavailable_metrics": ["client streaming TTFT", "true DRAM bandwidth", "SM occupancy",
                                    "per-request phase attribution under concurrent scheduling"],
            "instrumentation": "host wall clock only",
        }
        if not concurrent:
            _synchronize(backend, config)
        if not detailed:
            telemetry_report["token_timing_unavailable_reason"] = "Detailed instrumentation disabled for comparable formal timing."
    else:
        with RequestTelemetry(backend, config, _torch(backend)) as telemetry:
            result = getattr(backend, operation)(**arguments)
        _synchronize(backend, config)
        usage = result.get("usage") or {}
        outputs = len(result["token_ids"]) if isinstance(result.get("token_ids"), list) else usage.get("completion_tokens")
        telemetry_report = telemetry.report(outputs)
    elapsed = time.perf_counter() - started
    usage = result.get("usage") or {}
    inputs = len(result["input_ids"]) if isinstance(result.get("input_ids"), list) else usage.get("prompt_tokens")
    outputs = len(result["token_ids"]) if isinstance(result.get("token_ids"), list) else usage.get("completion_tokens")
    observation = {
        "kind": "host_request_timing", "clock": "perf_counter", "operation": operation,
        "backend_wall_seconds": elapsed, "queue_seconds": queue_seconds,
        "input_tokens": inputs, "output_tokens": outputs,
        "output_tokens_per_second": outputs / elapsed if isinstance(outputs, int) and elapsed > 0 else None,
        "ttft_seconds": None, "tpot_seconds": None,
        "timing_scope": "backend call including tokenization, generation, decoding and boundary synchronization; excludes HTTP",
        "token_count_scope": "backend IDs/usage; independently checked against frozen reference in formal evaluation",
        "streaming": "buffered; internal token readiness is not client streaming TTFT",
        "batch_size": result.get("batch_size"),
        "batching_mode": result.get("batching_mode"),
        "batch_queue_seconds": result.get("batch_queue_seconds"),
    }
    if operation == "chat_completion":
        details = usage.get("completion_tokens_details") or {}
        observation["finish_reason"] = result.get("finish_reason")
        observation["reasoning_tokens"] = details.get("reasoning_tokens")
        observation["visible_tokens"] = details.get("visible_tokens")
        observation["thinking_complete"] = (result.get("thinking") or {}).get("complete")
    observation.update(telemetry_report)
    return result, observation


def profile_generate(backend, config, prompt, max_new_tokens=16):
    """Same generate path, diagnostic run only. No arbitrary profiler code/paths."""
    if not isinstance(prompt, str) or len(prompt) > 65536:
        raise ValueError("Profile prompt exceeds diagnostic limit")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 128:
        raise ValueError("Profile output must be between 1 and 128 tokens")
    arguments = {"prompt": prompt, "max_new_tokens": max_new_tokens, "fixed_length": True}
    # Separate warmup, excluded from the diagnostic record and all formal samples.
    observe_call(backend, config, "generate", arguments)
    _, normal_timing = observe_call(backend, config, "generate", arguments)
    torch = _torch(backend)
    model = getattr(backend, "MODEL", None)
    phases, handles, spans, module_input_signatures = [], [], [], {}
    if torch is not None and model is not None and hasattr(model, "register_forward_pre_hook"):
        def before(module, args):
            name = "vibeforge.prefill" if not phases else "vibeforge.decode"
            phases.append(name)
            span = torch.profiler.record_function(name)
            span.__enter__()
            spans.append(span)

        def after(module, args, result):
            if spans:
                spans.pop().__exit__(None, None, None)

        handles.append(model.register_forward_pre_hook(before))
        handles.append(model.register_forward_hook(after, always_call=True))
        # Coarse sibling module groups, not thousands of Linear hooks. Nested
        # module intervals are inclusive and must never be summed with wall time.
        def install_group(module, group):
            active = []
            def begin(module, args, kwargs):
                signatures = module_input_signatures.setdefault(group, [])
                for signature in _tensor_signatures((args, kwargs)):
                    if signature not in signatures and len(signatures) < 8:
                        signatures.append(signature)
                span = torch.profiler.record_function("vibeforge.module." + group)
                span.__enter__()
                active.append(span)
            def end(module, args, output):
                if active:
                    active.pop().__exit__(None, None, None)
            handles.append(module.register_forward_pre_hook(begin, with_kwargs=True))
            handles.append(module.register_forward_hook(end, always_call=True))
        for name, module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in {"linear_attn", "self_attn", "mlp", "lm_head"}:
                group = {"linear_attn": "linear_attention", "self_attn": "full_attention"}.get(leaf, leaf)
                install_group(module, group)
    use_torch = torch is not None and model is not None
    if use_torch:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if str(config.get("device", "cpu")).startswith(("cuda", "rocm")):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        # Full operator-level shape grouping retains a very large event graph on
        # deep autoregressive models. Coarse module hooks above capture the
        # decision-relevant shapes at bounded memory cost.
        profiler = torch.profiler.profile(activities=activities, record_shapes=False, with_stack=False)
    else:
        profiler = cProfile.Profile()
    try:
        if use_torch:
            with profiler:
                result, timing = observe_call(backend, config, "generate", arguments)
        else:
            profiler.enable()
            try:
                result, timing = observe_call(backend, config, "generate", arguments)
            finally:
                profiler.disable()
    finally:
        for handle in handles:
            handle.remove()
        while spans:
            spans.pop().__exit__(None, None, None)
    if use_torch:
        all_events = profiler.key_averages()
        phase_rows, cpu_rows, device_rows, device_operator_rows = _torch_summaries(all_events)
        profiler_name = "torch.profiler"
    else:
        rows = [{"name": f"{entry.code.co_filename}:{entry.code.co_firstlineno}:{entry.code.co_name}"
                 if hasattr(entry.code, "co_filename") else str(entry.code),
                 "calls": entry.callcount, "self_cpu_seconds": entry.inlinetime,
                 "cpu_total_seconds": entry.totaltime}
                for entry in profiler.getstats()]
        cpu_rows = sorted(rows, key=lambda row: row["self_cpu_seconds"], reverse=True)[:15]
        phase_rows, device_rows, device_operator_rows, profiler_name = [], [], [], "cProfile (Python/CPU only)"
    measurements = {}
    def add(key, value, unit, scope):
        if isinstance(value, (int, float)):
            measurements[key] = {"value": value, "unit": unit, "scope": scope}
    for key in ("backend_wall_seconds", "ttft_seconds", "tpot_seconds", "queue_seconds"):
        add("normal." + key, normal_timing.get(key), "seconds", "ordinary inference; excludes profiler run")
    for key, value in normal_timing.get("phases", {}).items():
        add("normal.phases." + key, value, "seconds" if key.endswith("seconds") else "count", "ordinary inference")
    for row in phase_rows:
        for key in ("cpu_total_seconds", "device_total_seconds", "calls"):
            add(row["name"] + "." + key, row.get(key), "count" if key == "calls" else "seconds", "instrumented inclusive module/phase; overlapping, not wall-time fractions")
    for row in cpu_rows:
        add("operator." + row["name"] + ".calls", row["calls"], "count", "diagnostic operator call count")
    launch = next((row for row in cpu_rows if row["name"] == "cudaLaunchKernel"), None)
    diagnostics = {
        "kernel_launches": launch.get("calls") if launch else None,
        "kernel_launches_per_output_token": (
            launch.get("calls") / len(result["token_ids"])
            if launch and result.get("token_ids") else None),
        "profiled_device_kernel_families": len(device_rows),
        "top_device_kernel_share": (
            device_rows[0]["self_device_seconds"] /
            sum(row["self_device_seconds"] for row in device_rows)
            if device_rows and sum(row["self_device_seconds"] for row in device_rows) > 0 else None),
        "shape_capture": "bounded_module_forward_hooks" if use_torch else "unavailable",
        "module_input_signatures": module_input_signatures,
        "scope": "ranked profiler rows; top-device share covers retained top rows, not all GPU time",
    }
    return {"schema_version": 4, "kind": "host_inference_profile", "profiler": profiler_name,
            "normal_request": normal_timing, "timing": timing,
            "device": config.get("device", "cpu"), "precision": config.get("precision"),
            "profile_max_new_tokens": max_new_tokens,
            "top_cpu_ops": cpu_rows, "top_device_ops": device_rows,
            "top_device_operator_ops": device_operator_rows,
            "implementation_index": {
                "symbols": sorted(implementation_sources(backend))[:80],
                "scope": "Loaded backend/model Python definitions; availability does not imply a measured bottleneck.",
                "read_tool": "read_implementation(symbol=...) returns source and its hash without executing edits",
            },
            "diagnostics": diagnostics,
            "profiler_scopes": {"phases": "CPU record_function inclusive spans with attributed device kernels; CUDA user-annotation duplicates excluded",
                                "top_device_ops": "device kernel/memcpy events, excluding user annotations",
                                "top_device_operator_ops": "CPU operator self device attribution; overlaps kernel list, never add the two"},
            "phases": phase_rows, "output_tokens": len(result["token_ids"]),
            "measurements": measurements,
            "limitations": ["diagnostic instrumentation overhead; excluded from admission samples",
                            "CPU/device times overlap and must not be summed as wall time",
                            "prefill means first model forward; decode means subsequent forwards; missing hooks mean unavailable",
                            "hotspots support hypotheses, not causal proof; verify with paired experiments"]}
