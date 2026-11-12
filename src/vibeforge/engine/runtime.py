"""Host inspection of the loaded model, rather than requested configuration."""

from collections import Counter
from importlib import metadata
import sys


def runtime_manifest(backend, config, precision=None):
    model = getattr(backend, "MODEL", None)
    implementations = {}
    if model is not None:
        for name in ("config",):
            root = getattr(model, name, None)
            for label, value in (("model", root), ("text", getattr(root, "text_config", None))):
                if value is not None:
                    implementations[label] = getattr(value, "_attn_implementation", None)
    modules = Counter(type(module).__name__ for module in model.modules()) if model is not None else Counter()
    kernel_bindings = {}
    if model is not None:
        for module in model.modules():
            namespace = sys.modules.get(type(module).__module__)
            for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
                if namespace is not None and hasattr(namespace, name):
                    function = getattr(namespace, name)
                    kernel_bindings[name] = {"available": callable(function),
                        "implementation": getattr(function, "__module__", None),
                        "scope": "loaded symbol availability; execution must be confirmed in operator profile"}
    packages = {}
    for name in ("torch", "transformers", "bitsandbytes", "flash-attn", "flash-linear-attention", "causal-conv1d"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    for name, binding in kernel_bindings.items():
        dependency = "causal-conv1d" if name.startswith("causal_conv1d") else "flash-linear-attention"
        binding.update(optimized_dependency=dependency,
                       optimized_dependency_present=bool(packages[dependency]),
                       scope="A callable may be a Transformers fallback wrapper; dependency presence and a loaded symbol do not prove optimized kernel execution.")
    generation_functions = [getattr(backend, name, None) for name in (
        "generate", "_generate_direct", "_generate_batch")]
    uses_transformers_generate = any(
        code and "MODEL" in code.co_names and "generate" in code.co_names
        for code in (getattr(function, "__code__", None) for function in generation_functions))
    scheduler = config.get("request_scheduler", "serialized_lock")
    path = ("transformers_microbatch_generate" if scheduler == "microbatch" else
            "transformers_generate") if model is not None and uses_transformers_generate else "custom_or_unverified"
    microbatch_supported = bool(getattr(backend, "MICROBATCH_SUPPORTED", False))
    return {
        "schema_version": 2, "model_class": type(model).__name__ if model is not None else None,
        "execution_path": path,
        "request_scheduler": scheduler,
        "continuous_batching": False,
        "batching_mode": "static_microbatch" if scheduler == "microbatch" and microbatch_supported else "none",
        "scheduler_capabilities": {
            "available": ["serialized_lock", "concurrent_requests", *(["microbatch"] if microbatch_supported else [])],
            "microbatch_max_size": config.get("microbatch_max_size") if scheduler == "microbatch" else None,
            "paged_kv_cache": False,
        },
        "attention_implementations": implementations,
        "module_classes": dict(modules), "packages": packages,
        "optional_kernel_bindings": kernel_bindings,
        "generation_defaults": {key: getattr(getattr(model, "generation_config", None), key, None)
                                for key in ("use_cache", "cache_implementation", "do_sample")},
        "generation_defaults_scope": "Model defaults only; the backend may override them. Consult host_observation.observed_generate_kwargs for arguments actually passed at request time.",
        "model_dtype": str(getattr(model, "dtype", "unavailable")),
        "model_device": str(getattr(model, "device", config.get("device", "cpu"))),
        "runtime_precision": precision,
        "requested": {key: config.get(key) for key in (
            "attn_implementation", "use_cache", "cache_implementation", "request_scheduler",
            "microbatch_max_size", "microbatch_wait_ms", "precision")},
        "limits": {key: config.get(key) for key in ("max_context", "max_new_tokens")},
        "compatibility": {
            "paged_attention": {"supported": False, "reason": "Requires a paged-cache allocator and a verified continuous-batching adapter; concurrent HTTP admission alone is insufficient."},
            "flash_attention_2": {"supported": bool(packages["flash-attn"]) and str(config.get("device", "cpu")).startswith("cuda"),
                                  "reason": "Also requires compatible model, CUDA architecture and compute dtype; startup is the final compatibility check."},
        },
        "limitations": ["Resolved model configuration is not proof that every layer executed the same kernel.",
                        "Optional package presence is not proof that its optimized kernels ran; consult the operator profile."]}
