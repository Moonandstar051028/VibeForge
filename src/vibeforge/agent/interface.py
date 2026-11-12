"""Shared model-facing context and capability descriptions for all Agent adapters."""

import copy


EVIDENCE_TOOLS = ("list_evidence", "read_evidence", "search_history")


def agent_capabilities(context, *, analysis_only=False, smoke=False):
    """Describe the host capabilities independently of the selected Agent framework."""
    if smoke:
        tools = ["record_result"]
    else:
        tools = [*EVIDENCE_TOOLS, "record_result"]
        if context.get("telemetry_session") and context.get("engine_url"):
            tools.append("read_telemetry")
        profiling = (context.get("loop_protocol", {}).get("profiling") or {})
        if (context.get("candidate_contract") and context.get("engine_url") and not analysis_only
                and profiling.get("enabled", True)):
            tools.extend(["profile_inference", "read_implementation"])
    return {
        "schema_version": 1,
        "context_contract": "vibeforge-agent-context-v4",
        "tools": tools,
        "evidence_paths_are_campaign_relative": True,
        "profile_results_are_diagnostic_only": True,
    }


def bounded_context_view(context, *, analysis_only=False, smoke=False):
    """Return the common bounded first view used by Qwen-Agent and Codex."""
    observation = context.get("performance_observation") or {}
    keys = (
        "optimization_brief", "context_schema_version", "campaign_id", "iteration", "stable_ref",
        "requirements", "model", "model_features", "hardware", "environment",
        "workload", "workload_roles", "precision", "optimization", "modification_surfaces", "protocol",
        "loop_protocol", "effective_runtime", "previous", "previous_analysis",
        "history", "history_total", "older_history_summary", "experience_knowledge",
        "history_retrieval", "memory_index", "knowledge",
        "transfer_hints",
        "archived_campaign_memory", "archived_candidate_results",
        "archived_experience_knowledge", "candidate_contract",
        "telemetry_session", "candidate_feedback", "refinement", "refinement_target",
        "candidate_engine_config", "instructions",
    )
    view = {key: copy.deepcopy(context[key]) for key in keys if key in context}
    if view.get("optimization_brief"):
        for key in ("model_features", "hardware", "environment", "effective_runtime"):
            view.pop(key, None)
        view["full_context_retrieval"] = (
            "Full model, hardware, environment and runtime records remain in .vibeforge/context.json.")
    if not (view.get("older_history_summary") or {}).get("attempts"):
        view.pop("older_history_summary", None)
    for key in ("experience_knowledge", "archived_experience_knowledge"):
        experience = view.get(key) or {}
        if not experience.get("verified_strategies") and not experience.get("counterexamples"):
            view.pop(key, None)
    environment = view.get("environment")
    if isinstance(environment, dict):
        inference = environment.get("inference") or {}
        packages = inference.get("packages") or {}
        if packages:
            relevant = {
                "torch", "transformers", "bitsandbytes", "accelerate", "triton",
                "vllm", "xformers", "flash-attn", "flash-linear-attention",
                "causal-conv1d", "safetensors",
            }
            inference["packages"] = {
                name: version for name, version in packages.items()
                if name.casefold().replace("_", "-") in relevant
            }
            inference["package_count"] = len(packages)
            inference["full_package_inventory"] = (
                "Available in .vibeforge/context.json; query only when a dependency matters.")
    contract = view.get("candidate_contract")
    if isinstance(contract, dict):
        view["candidate_contract"] = {
            "version": contract.get("version"),
            "observation_paths": contract.get("observation_paths", []),
            "history_refs": contract.get("history_refs", [])[-50:],
            "tested_candidates": contract.get("tested_candidates", [])[-12:],
            "full_contract": "Available in .vibeforge/context.json.",
        }
    view["performance_observation"] = {
        key: copy.deepcopy(observation[key]) for key in (
            "kind", "profile_id", "prompt_index", "normal_request", "measurements",
            "phases", "top_cpu_ops", "top_device_ops", "top_device_operator_ops",
            "runtime", "profiler", "profiler_scopes", "profile_matrix", "limitations", "observed_at",
            "benefit_bounds", "diagnostics", "concurrency_diagnostics",
        ) if key in observation
    }
    view["agent_interface"] = agent_capabilities(
        context, analysis_only=analysis_only, smoke=smoke)
    return view
