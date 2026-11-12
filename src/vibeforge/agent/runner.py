"""Run the selected Agent framework against the campaign's local inference server."""

import copy
import json
import tempfile
import time
import uuid
import difflib
from contextlib import closing
from pathlib import Path

from vibeforge.agent.analysis import analysis_attribution_error, is_placeholder_analysis, performance_analysis_error
from vibeforge.agent.backend_policy import (
    OPTIONAL_BACKEND_PACKAGES,
    QWEN35_ATTENTION_BACKENDS,
    TRANSFORMERS_CACHE_IMPLEMENTATIONS,
    TRANSFORMERS_ATTENTION_BACKENDS,
    attention_backend_issues,
    rejected_attention_backend_families,
)
from vibeforge.agent.tools import AgentTools
from vibeforge.agent.interface import agent_capabilities
from vibeforge.agent.profile_tool import profile_inference, read_implementation
from vibeforge.agent.candidate import candidate_gate
from vibeforge.agent.memory import reflection_error
from vibeforge.engine.client import EngineClient
from vibeforge.observability.logs import append_record
from vibeforge.storage.atomic import digest, read_json, write_json
from vibeforge.versioning.artifacts import inventory


def candidate_edit_summary(parent, workspace):
    """Describe all candidate edits, including configuration and helper modules."""
    parent, workspace = Path(parent), Path(workspace)
    before, after = inventory(parent), inventory(workspace)
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    patches = []
    remaining = 8000
    for name in changed:
        if remaining <= 0 or Path(name).suffix not in {".py", ".json", ".cu", ".cpp", ".c"}:
            continue
        original, candidate = parent / name, workspace / name
        a = original.read_text(errors="replace").splitlines(True) if original.is_file() else []
        b = candidate.read_text(errors="replace").splitlines(True) if candidate.is_file() else []
        patch = "".join(difflib.unified_diff(a, b, fromfile='stable/' + name, tofile='candidate/' + name))
        patches.append(patch[:remaining])
        remaining -= len(patches[-1])
    return {"workspace_changed": bool(changed), "changed_files": changed[:50], "candidate_diff": "\n".join(patches)}


TOOL_SCHEMAS = {
    "read_implementation": ("Read loaded model/helper source. Omit symbol to list names; then page one function. Read-only, not performance evidence.",
                            {"symbol": "string", "offset": "integer", "limit": "integer"}, []),
    "read_file": ("Read a candidate-relative file; offset and limit are character counts.",
                  {"path": "string", "offset": "integer", "limit": "integer"}, ["path"]),
    "replace_text": ("Replace one exact, unique text span in a candidate file.",
                     {"path": "string", "old": "string", "new": "string"}, ["path", "old", "new"]),
    "write_file": ("Write a file inside the candidate workspace.",
                   {"path": "string", "content": "string"}, ["path", "content"]),
    "list_files": ("List files in the candidate workspace.", {}, []),
    "check_python": ("Parse a Python file to check its syntax.", {"path": "string"}, ["path"]),
    "read_evidence": ("Read a campaign evidence file by evidence-root-relative path.",
                      {"path": "string", "offset": "integer", "limit": "integer"}, ["path"]),
    "list_evidence": ("List files beneath a campaign evidence subdirectory.",
                       {"path": "string", "offset": "integer"}, ["path"]),
    "search_history": ("Search recorded hypotheses, decisions and previous analyses; bounded retrieval.",
                       {"query": "string", "limit": "integer"}, ["query"]),
    "profile_inference": ("Run bounded host profiling on a frozen workload prompt using the stable engine; diagnostic only (at most twice per session).",
                          {"prompt_index": "integer", "profile_id": "string"}, ["prompt_index"]),
    "read_telemetry": ("Read the bounded summary of ordinary inference requests made by this Agent session; diagnostic only.",
                       {}, []),
    "report_blocked": ("Record why no justified candidate can be submitted; this is not an experiment.",
                       {"reason": "string", "previous_analysis": "string", "reflection": "object"}, ["reason"]),
    "submit_candidate": ("Submit the hypothesis, previous-result analysis, and change summary.", {
        "hypothesis": "string", "previous_analysis": "string", "summary": "string",
        "experiment_plan": {"type": "object", "properties": {
            **{key: {"type": "string"} for key in ("measured_bottleneck", "cause", "change", "metric", "direction", "validation")},
            "expected_change_pct": {"type": "number"},
            "bottleneck_metric": {"type": "string"},
            "evidence_paths": {"type": "array", "items": {"type": "string"}},
            "history_refs": {"type": "array", "items": {}},
            "bottleneck_reference": {"type": "object", "properties": {
                "evidence_path": {"type": "string"}, "metric": {"type": "string"},
                "value": {"type": "number"}, "unit": {"type": "string"}},
                "required": ["evidence_path", "metric", "value", "unit"]},
            "target_profile_ids": {"type": "array", "items": {"type": "string"}},
            "runtime_preconditions": {"type": "array", "items": {"type": "object", "properties": {
                "field": {"type": "string"}, "equals": {}}, "required": ["field", "equals"]}},
            "expected_bottleneck_change_pct": {"type": "number"},
            "bottleneck_direction": {"type": "string"}, "expected_change_basis": {"type": "string"},
        }, "required": ["measured_bottleneck", "evidence_paths", "cause", "change", "metric", "direction", "expected_change_pct", "validation", "history_refs"]},
        "reflection": {"type": "object", "properties": {
            "iteration": {"type": "integer"}, "hypothesis_status": {"type": "string"},
            "explanation": {"type": "string"}, "next_action": {"type": "string"},
            "evidence_paths": {"type": "array", "items": {"type": "string"}},
        }},
    }, ["hypothesis"]),
}
TOOL_SCHEMAS["report_blocked"][1]["reflection"] = TOOL_SCHEMAS["submit_candidate"][1]["reflection"]

def _make_tool(name, callback, *, required_fields=(), plan_version=None, trace=None):
    from jsonschema.exceptions import ValidationError
    from qwen_agent.tools.base import BaseTool

    description, properties, required = TOOL_SCHEMAS[name]
    parameters = {
        "type": "object",
        "properties": {key: copy.deepcopy(value) if isinstance(value, dict) else {"type": value}
                       for key, value in properties.items()},
        "required": list(dict.fromkeys([*required, *required_fields])),
    }
    if plan_version is not None:
        description += (" Pass the complete plan as ONE nested JSON object named experiment_plan; "
                        "plan fields at the top level or JSON-encoded strings are invalid.")
        plan = parameters["properties"]["experiment_plan"]
        plan["description"] = "Nested experiment plan object; preserve JSON number, array and object types."
        if plan_version >= 3:
            plan["properties"]["history_refs"]["items"] = {"type": "string"}
            plan["required"] = ["measured_bottleneck", "cause", "change", "expected_change_pct",
                                "validation", "history_refs"]
            plan["description"] += (" Controller fills bound evidence and rejects predictions above the "
                                    "measured phase's Amdahl limit. Optionally pass "
                                    "bottleneck_metric as an exact measurements key. The controller owns "
                                    "metric and direction from the frozen campaign objective; omit them.")
        elif plan_version >= 2:
            plan["properties"]["history_refs"]["items"] = {"type": "integer"}
            plan["required"] += ["bottleneck_reference", "target_profile_ids", "runtime_preconditions",
                                 "expected_bottleneck_change_pct", "bottleneck_direction", "expected_change_basis"]
        else:
            plan["properties"]["history_refs"]["items"] = {"type": "integer"}

    def call(self, params, **kwargs):
        try:
            arguments = self._verify_json_format_args(params, strict_json=True)
        except ValidationError as error:
            if name not in {"submit_candidate", "report_blocked"}:
                raise
            result = {"submitted": False, "error": {"reason": "invalid_tool_arguments",
                      "path": list(error.absolute_path), "message": error.message},
                      "next_action": "Follow this session's tool schema. For a candidate, pass experiment_plan as a nested JSON object; do not flatten its fields or encode arrays and objects as strings."}
            if trace:
                append_record(trace, {"tool": name, "error": error.message, "validation_error": result["error"], "result": result})
            return json.dumps(result, ensure_ascii=False)
        result = callback(arguments)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    tool_type = type(
        f"VibeForge_{name}",
        (BaseTool,),
        {"name": name, "description": description, "parameters": parameters, "call": call},
    )
    return tool_type()


def _assistant_tools(workspace, trace, context, submission, analysis_only=False, smoke=False, submit_only=False,
                     attempt_state=None):
    tools = AgentTools(workspace, trace, context.get("evidence_root"))
    names = ["submit_candidate"] if smoke or submit_only else ["read_file", "list_files", "submit_candidate"]
    if not analysis_only and not smoke and not submit_only:
        names.extend(["replace_text", "write_file", "check_python"])
    if not smoke and not submit_only and context.get("evidence_root"):
        names.extend(["list_evidence", "read_evidence", "search_history"])
    if not smoke and not submit_only and context.get("telemetry_session") and context.get("engine_url"):
        names.append("read_telemetry")
    if context.get("candidate_contract") and not smoke and not analysis_only and not submit_only:
        names.append("report_blocked")
        profiling = (context.get("loop_protocol", {}).get("profiling") or {})
        if context.get("engine_url") and profiling.get("enabled", True):
            names.extend(["profile_inference", "read_implementation"])
    profile_calls = []
    telemetry_calls = []

    def telemetry_model_view(report):
        latest = report.get("latest_request") or {}
        return {
            key: report.get(key) for key in (
                "schema_version", "kind", "request_count", "dropped_requests", "operations",
                "protocols", "input_tokens", "output_tokens", "backend_wall_seconds",
                "aggregate_output_tokens_per_second", "latency_seconds", "queue_seconds",
                "ttft_seconds", "tpot_seconds", "phase_seconds", "admission_role",
            ) if report.get(key) is not None
        } | {"latest_request": {
            key: latest.get(key) for key in (
                "protocol", "operation", "backend_wall_seconds", "queue_seconds", "input_tokens",
                "output_tokens", "output_tokens_per_second", "ttft_seconds", "tpot_seconds",
                "finish_reason", "reasoning_tokens", "visible_tokens", "thinking_complete", "phases",
            ) if latest.get(key) is not None
        }}

    def execute(name):
        def callback(arguments):
            if name == "read_implementation":
                return read_implementation(context, trace, arguments)
            if name == "profile_inference":
                result = profile_inference(context, trace, arguments, profile_calls)
                result["profile"] = _model_context(
                    {"performance_observation": result["profile"]})["performance_observation"]
                return result
            if name == "read_telemetry":
                if len(telemetry_calls) >= 4:
                    raise ValueError("Telemetry read budget exhausted")
                telemetry_calls.append(time.time())
                report = EngineClient(
                    context["engine_url"], context["agent"].get("request_timeout", 180)
                ).telemetry_report(context["telemetry_session"]["session_id"])
                relative = context["telemetry_session"]["evidence_path"].replace(
                    ".json", f"-live-{len(telemetry_calls)}.json")
                write_json(Path(context["evidence_root"]) / relative, report)
                append_record(trace, {"tool": name, "arguments": arguments,
                                      "result_path": relative,
                                      "request_count": report["request_count"]})
                return {"evidence_path": relative, "telemetry": telemetry_model_view(report),
                        "full_report": "Use read_evidence with the evidence_path only if the aggregate is insufficient."}
            if name not in {"submit_candidate", "report_blocked"}:
                return tools.execute(name, arguments)
            try:
                hypothesis = arguments.get("hypothesis", arguments.get("reason", "")).strip()
                if not hypothesis:
                    raise ValueError("Submission requires a hypothesis")
                previous_analysis = arguments.get("previous_analysis", "").strip()
                if (context.get("previous") or {}).get("analysis_pending"):
                    if not previous_analysis:
                        raise ValueError("Analyze the previous recorded result before submitting another candidate")
                    if is_placeholder_analysis(previous_analysis):
                        raise ValueError("The previous analysis is a placeholder; analyze its recorded decision before submitting")
                attribution_error = analysis_attribution_error(context.get("previous"), previous_analysis)
                if attribution_error:
                    result = {"submitted": False, "error": attribution_error}
                    append_record(trace, {"tool": name, "arguments": arguments, "result": result})
                    return result
                evidence_error = performance_analysis_error(context.get("previous"), previous_analysis, arguments.get("reflection"))
                if evidence_error:
                    result = {"submitted": False, "error": evidence_error}
                    append_record(trace, {"tool": name, "arguments": arguments, "result": result})
                    return result
                if context.get("candidate_contract"):
                    error = reflection_error(context.get("previous"), arguments.get("reflection"), context["evidence_root"])
                    if error:
                        result = {"submitted": False, "error": error}
                        append_record(trace, {"tool": name, "arguments": arguments, "result": result})
                        return result
                if name == "report_blocked":
                    submission.update(hypothesis=hypothesis, blocked=True, blocked_reason=hypothesis,
                                      previous_analysis=previous_analysis, reflection=arguments.get("reflection"), complete=True)
                    append_record(trace, {"tool": name, "arguments": arguments, "result": {"blocked": True}})
                    return {"blocked": True}
                issues = attention_backend_issues(workspace, context) if not smoke and not analysis_only else []
                if issues:
                    backend, reason = issues[0]
                    message = (f"Candidate selects {backend}, which {reason}. Choose a backend supported by the "
                               "configured model and captured environment, or report a blocked proposal.")
                    append_record(trace, {"tool": name, "arguments": arguments,
                                          "result": {"submitted": False, "error": message}})
                    return {"submitted": False, "error": message}
                gate = candidate_gate(workspace, context, arguments.get("experiment_plan")) if not smoke and not analysis_only else {"valid": True}
                if not gate["valid"]:
                    result = {"submitted": False, "error": gate, "next_action": "Fix the proposal or call report_blocked; no-change is not an experiment."}
                    if attempt_state is not None:
                        attempt_state["rejected_submission_count"] = attempt_state.get("rejected_submission_count", 0) + 1
                        attempt_state["last_rejected_arguments"] = arguments
                        attempt_state["last_rejection"] = result
                    append_record(trace, {"tool": name, "arguments": arguments, "result": result})
                    return result
                if gate.get("experiment_plan"):
                    arguments["experiment_plan"] = gate["experiment_plan"]
                submission.update({
                    "hypothesis": hypothesis,
                    "previous_analysis": previous_analysis,
                    "analysis_iteration": (context.get("previous") or {}).get("iteration"),
                    "summary": arguments.get("summary", "").strip(),
                    "experiment_plan": arguments.get("experiment_plan"),
                    "reflection": arguments.get("reflection"),
                    "complete": True,
                })
                append_record(trace, {"tool": name, "arguments": arguments, "result": {"submitted": True}})
                return {"submitted": True}
            except Exception as error:
                append_record(trace, {"tool": name, "arguments": arguments, "error": str(error)})
                raise
        return callback

    registered = []
    for name in names:
        required_fields = []
        plan_version = None
        if not smoke:
            if name == "submit_candidate" and context.get("candidate_contract") and not analysis_only:
                required_fields.append("experiment_plan")
                plan_version = context["candidate_contract"].get("version", 1)
            if name in {"submit_candidate", "report_blocked"} and (context.get("previous") or {}).get("analysis_pending"):
                required_fields.append("previous_analysis")
            if name in {"submit_candidate", "report_blocked"} and context.get("candidate_contract") and (context.get("previous") or {}).get("analysis_pending"):
                required_fields.append("reflection")
        registered.append(_make_tool(name, execute(name), required_fields=required_fields,
                                     plan_version=plan_version, trace=trace))
    return registered


def _previous_analysis_for_recovery(context):
    previous = context.get("previous")
    if not previous:
        return "This is the initial candidate; no previous experiment is available to analyze."
    decision = previous.get("decision") or {}
    reason = decision.get("reason", "unknown")
    gain = decision.get("gain")
    if isinstance(gain, (int, float)):
        lower, upper = decision.get("lower"), decision.get("upper")
        interval = (f" Its confidence interval was [{lower:.4f}, {upper:.4f}]."
                    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) else "")
        return (f"Iteration {previous.get('iteration')} measured a {gain:.2%} change in "
                f"{decision.get('objective', 'the objective')} and was rejected for {reason}.{interval} "
                "This result does not support retaining that candidate.")
    return (f"Iteration {previous.get('iteration')} was rejected for {reason} and did not produce a "
            "usable performance estimate. Treat it as inconclusive and choose a distinct hypothesis.")


def _recoverable_model_error(error):
    message = str(error).casefold()
    return ("malformed or truncated qwen tool call" in message
            or "request exceeds configured context" in message)


def _agent_input_budget(context, max_tokens):
    limit = int(context.get("model", {}).get("max_context", 4096))
    # Keep a transport reserve even when the active engine offers exact counting.
    budget = limit - max_tokens - max(1024, limit // 5)
    if budget <= 0:
        raise ValueError("The engine context is too small for the configured agent output budget")
    return budget


def _model_context(context, analysis_only=False):
    """Keep growing admission indexes on the host, outside the model prompt."""
    value = json.loads(json.dumps(context))
    if value.get("optimization_brief") and not analysis_only:
        for key in ("model_features", "hardware", "effective_runtime"):
            value.pop(key, None)
        value["full_context_retrieval"] = (
            "Full model, hardware, environment and runtime records remain in this iteration's context.json.")
    if not (value.get("older_history_summary") or {}).get("attempts"):
        value.pop("older_history_summary", None)
    for key in ("experience_knowledge", "archived_experience_knowledge"):
        experience = value.get(key) or {}
        if not experience.get("verified_strategies") and not experience.get("counterexamples"):
            value.pop(key, None)
    value["agent_interface"] = agent_capabilities(context, analysis_only=analysis_only)
    feedback = value.get("candidate_feedback")
    if isinstance(feedback, dict):
        correctness = feedback.get("correctness_precheck") or {}
        runtime_observation = correctness.get("runtime_observation") or {}
        normal = runtime_observation.get("normal_request") or {}
        profile_reports = feedback.get("profile_reports") or []
        candidate_normal = ((profile_reports[0].get("normal_request") or {})
                            if profile_reports and isinstance(profile_reports[0], dict) else {})
        value["candidate_feedback"] = {
            "candidate": feedback.get("candidate"),
            "correctness_precheck": {
                "passed": correctness.get("passed"),
                "checks": [{key: check.get(key) for key in ("case", "passed", "error")
                            if check.get(key) is not None}
                           for check in correctness.get("checks", [])],
            },
            "mechanism": feedback.get("mechanism"),
            "runtime_effect": {key: normal.get(key) for key in (
                "observed_cache_class", "backend_wall_seconds", "output_tokens_per_second",
                "ttft_seconds", "tpot_seconds") if normal.get(key) is not None},
            "candidate_profile_request": {key: candidate_normal.get(key) for key in (
                "backend_wall_seconds", "output_tokens_per_second", "ttft_seconds", "tpot_seconds")
                if candidate_normal.get(key) is not None},
            "profile_id": feedback.get("profile_id"),
            "prompt_index": feedback.get("prompt_index"),
            "feedback_state": feedback.get("feedback_state"),
            "recommended_action": feedback.get("recommended_action"),
            "needs_revision": feedback.get("needs_revision"),
            "evidence_path": value.get("candidate_feedback_path"),
            "limitations": feedback.get("limitations", []),
        }
    loop = value.get("loop_protocol")
    if isinstance(loop, dict):
        # The complete frozen budgets remain in context.json.  The model only
        # needs mode and availability facts when choosing an experiment.
        value["loop_protocol"] = {
            "telemetry": {key: loop.get("telemetry", {}).get(key)
                          for key in ("enabled", "capture_agent_requests")},
            "profiling": {key: loop.get("profiling", {}).get(key)
                          for key in ("enabled", "coverage", "max_profiles_per_session")},
            "probe": {"enabled": loop.get("probe", {}).get("enabled", False)},
            "refinement": {key: loop.get("refinement", {}).get(key)
                           for key in ("enabled", "max_revisions")},
            "full_contract": "Frozen budgets and timeouts are available in context.json.",
        }
    if analysis_only:
        # Current stable source/profile can differ from the assigned historical
        # experiment and must not redirect a review into a new optimization.
        keys = ("campaign_id", "iteration", "previous", "previous_analysis", "protocol", "requirements",
                "pending_analyses", "history_total", "history_retrieval", "memory_index", "evidence_root")
        value = {key: value[key] for key in keys if key in value}
        if (value.get("previous") or {}).get("prior_analysis_validation_error"):
            # An invalidated review is evidence to audit, not trusted planning memory.
            value["previous"].pop("interpretation", None)
            value["previous"].pop("reflection", None)
            value.pop("previous_analysis", None)
            value["invalidated_review"] = "The old review failed evidence validation. Rebuild this review from recorded facts; original model text remains in runtime logs."
        value["instructions"] = "Review only context.previous; read its evidence and submit previous_analysis and reflection. Do not propose or edit a candidate."
        target = (value.get("previous") or {}).get("iteration")
        if isinstance(target, int):
            value["target_evidence_directory"] = f"iterations/{target:04d}"
            value["decision_evidence_path"] = f"iterations/{target:04d}/decision.json"
            correctness_path = Path(value.get("evidence_root", "")) / f"iterations/{target:04d}/correctness/report.json"
            if correctness_path.is_file():
                correctness = read_json(correctness_path)
                value["correctness_summary"] = {
                    "evidence_path": f"iterations/{target:04d}/correctness/report.json",
                    "passed": correctness.get("passed"),
                    "failure_summary": correctness.get("failure_summary"),
                    "failed_checks": [{key: check.get(key) for key in (
                        "case", "error", "generation_passed", "repeat_passed", "fixed_length",
                        "numerical", "continuation_numerical",
                    ) if key in check} for check in correctness.get("checks", []) if not check.get("passed")],
                }
    def compact_workload(workload):
        if not isinstance(workload, dict):
            return
        for profile in [workload, *workload.get("profiles", [])]:
            if "prompts" in profile:
                profile["prompts"] = [{"index": i, "characters": len(prompt), "preview": prompt[:160],
                                        "content_in_frozen_host_context": True} for i, prompt in enumerate(profile["prompts"])]
    compact_workload(value.get("workload"))
    environment = value.get("environment")
    if isinstance(environment, dict):
        inference = environment.get("inference") or {}
        packages = inference.get("packages") or {}
        keep = {name: version for name, version in packages.items() if name.casefold().replace("_", "-") in {
            "torch", "transformers", "bitsandbytes", "accelerate", "triton", "vllm", "xformers", "flash-attn"
        }}
        if packages:
            inference["packages"] = keep
            inference["package_count"] = len(packages)
            inference["full_package_inventory"] = "Read this iteration's context.json with read_evidence when needed."
    profile = value.get("performance_observation")
    if isinstance(profile, dict):
        # Full CUDA kernel names and duplicate request arrays can consume the
        # entire input budget. Keep numeric evidence intact; raw rows remain in
        # the host artifact and can be paged with read_evidence.
        profile.pop("workload", None)
        diagnostic = profile.pop("timing", None)
        if diagnostic:
            profile["diagnostic_wall_seconds"] = diagnostic.get("backend_wall_seconds")
        for key in ("top_cpu_ops", "top_device_ops", "top_device_operator_ops"):
            rows = profile.get(key)
            if isinstance(rows, list):
                selected = []
                for index, row in enumerate(rows[:5]):
                    row = dict(row, raw_index=index)
                    name = row.get("name")
                    if isinstance(name, str) and len(name) > 160:
                        row["name"] = name[:160] + "..."
                        row["name_truncated"] = True
                    selected.append(row)
                profile[key] = selected
        normal = profile.get("normal_request") or {}
        intervals = normal.pop("inter_token_seconds", None)
        if intervals:
            normal["inter_token_summary"] = {"count": len(intervals), "min_seconds": min(intervals),
                "max_seconds": max(intervals), "mean_seconds": sum(intervals) / len(intervals)}
        profile["context_summary"] = ("First five operator rows; long names are previews. Full measurements, "
            "runtime and module phases retained. Read the bound host evidence for remaining operators, "
            "full names, diagnostic timing, token intervals and frozen workload.")
    contract = context.get("candidate_contract")
    if contract and not analysis_only:
        value["candidate_contract"] = {
            "version": contract.get("version", 1),
            "observation_paths": contract.get("observation_paths", []),
            "allowed_history_refs": contract.get("history_refs", [])[-50:],
            "history_policy": ("Cite globally unique campaign_id:iteration values from allowed_history_refs. "
                               "Archived and current records share this namespace; the host validates the full index."),
        }
    previous = value.get("previous") or {}
    if previous.get("iteration") is not None:
        reason = (previous.get("decision") or {}).get("reason")
        verification = previous.get("experiment_verification") or {}
        decision_gain = (previous.get("decision") or {}).get("gain")
        verification_gain = verification.get("gain")
        formal_comparison = isinstance(decision_gain, (float, int)) or (
            isinstance(verification_gain, (float, int))
            and isinstance(verification.get("pairs"), int)
            and verification["pairs"] > 0
        )
        value["review_measurement_status"] = {
            "iteration": previous["iteration"],
            "formal_comparison_available": formal_comparison,
            "formal_comparison_source": (
                "decision" if isinstance(decision_gain, (float, int)) else
                "experiment_verification" if formal_comparison else None
            ),
            "decision_reason": reason,
            "observation_role": "observation.json is the stable parent before the candidate edit; it is not candidate performance",
            "observation_sections": {
                "normal_request": "stable parent ordinary inference, without profiler",
                "timing": "same stable parent inference WITH profiler; overhead changes timing; never a candidate result",
                "phases": "same stable parent instrumented module/phase events; inclusive and overlapping",
            },
            "candidate_diagnostic_path": f"iterations/{previous['iteration']:04d}/candidate-observation.json",
            "formal_comparison_directory": f"iterations/{previous['iteration']:04d}/benchmarks",
            "interpretation_rule": (
                "When formal comparison exists, cite its paired gain and interval from verification.json or the benchmark summary. "
                "A null decision gain can result from an earlier per-profile protection gate and does not erase a completed paired comparison. "
                "If no formal comparison exists, do not claim a measured candidate gain, regression or before/after performance. "
                "Correctness rejection does not measure throughput."
            ),
        }
    return value


def _fit_model_context(context, system, input_budget, reserve, tokenizer, analysis_only=False,
                       count_context=None, max_static_tokens=None):
    """Keep retrievable history details out of a prompt when they crowd out tools."""
    value = _model_context(context, analysis_only=analysis_only)
    steps = []
    def count():
        if count_context is not None:
            return count_context(value)
        return tokenizer.count_tokens(system) + tokenizer.count_tokens(json.dumps(value, ensure_ascii=False))
    # Tool schemas can legitimately consume more than 55% of a small context
    # window. The enforceable limit is the input budget minus an explicit
    # reserve for tool results; 55% is retained by the caller as telemetry.
    static_target = min(input_budget - reserve, max_static_tokens or input_budget)
    if count() > static_target:
        if value.get("history"):
            value["history"] = [{
                "iteration": row["iteration"], "candidate": row.get("candidate"),
                "hypothesis": (row.get("hypothesis") or "")[:240], "decision": row.get("decision"),
                "attention_backends": row.get("attention_backends", []),
                "evidence_path": row.get("evidence_path", f"iterations/{row['iteration']:04d}"),
                "hypothesis_status": (row.get("reflection") or {}).get("hypothesis_status"),
                "details": "Read memory.json, analysis.md and change.patch in evidence_path; search_history retrieves older records.",
            } for row in value["history"]]
            steps.append("history_details_retrievable")
        summary = value.get("older_history_summary") or {}
        ledger = summary.get("strategy_ledger")
        if isinstance(ledger, list) and len(ledger) > 12:
            summary["strategy_ledger"] = sorted(
                ledger, key=lambda row: row.get("attempts", 0), reverse=True)[:12]
            summary["strategy_ledger_omitted"] = len(ledger) - 12
            summary["complete_ledger"] = "Use search_history; every older record remains indexed."
            steps.append("older_strategy_ledger_retrievable")
        for key in ("experience_knowledge", "archived_experience_knowledge"):
            experience = value.get(key) or {}
            changed = False
            for category in ("verified_strategies", "counterexamples"):
                rows = experience.get(category)
                if isinstance(rows, list) and len(rows) > 8:
                    experience[category] = rows[-8:]
                    experience[category + "_omitted"] = len(rows) - 8
                    changed = True
            if changed:
                experience["complete_history"] = "Use search_history for all experience records."
                steps.append(key + "_retrievable")
        previous = value.get("previous")
        if previous:
            value["previous"] = {key: previous[key] for key in (
                "iteration", "parent", "candidate", "hypothesis", "decision", "error", "complete",
                "analysis_pending", "experiment_verification") if key in previous}
            if isinstance(previous.get("iteration"), int):
                value["previous_details_path"] = f"iterations/{previous['iteration']:04d}/memory.json"
            value["previous_analysis"] = value.get("previous_analysis", "")[:1800]
            steps.append("previous_details_retrievable")
    if count() > static_target and "source" in value:
        value.pop("source")
        value["source_retrieval"] = "Read current engine.py with read_file before editing; full source remains in the workspace."
        steps.append("source_via_read_file")
    if count() > static_target and "effective_runtime" in value:
        value.pop("effective_runtime")
        value["effective_runtime_retrieval"] = "Use performance_observation.runtime or read the bound observation artifact."
        steps.append("duplicate_runtime_removed")
    if count() > static_target:
        profile = value.get("performance_observation") or {}
        measurements = profile.get("measurements") or {}
        preferred = {key: measurements[key] for key in measurements if key.startswith("normal.")}
        if preferred and len(preferred) < len(measurements):
            profile["measurements"] = preferred
            profile["measurement_inventory"] = "Full measurements are in the bound observation artifact."
            steps.append("measurements_retrievable")
    if count() > static_target:
        profile = value.get("performance_observation") or {}
        normal = profile.get("normal_request") or {}
        if normal:
            profile["normal_request"] = {key: normal[key] for key in (
                "backend_wall_seconds", "queue_seconds", "input_tokens", "output_tokens",
                "output_tokens_per_second", "ttft_seconds", "tpot_seconds", "streaming",
                "token_timing_available", "token_timing_unavailable_reason", "observed_cache_class",
                "observed_generate_kwargs",
            ) if key in normal}
            profile["normal_request_retrieval"] = (
                "Full timing scope, phases and instrumentation are in the bound observation artifact.")
            steps.append("normal_request_details_retrievable")
        runtime = profile.get("runtime") or {}
        if runtime:
            profile["runtime"] = {key: runtime[key] for key in (
                "model_class", "execution_path", "request_scheduler", "continuous_batching",
                "batching_mode", "scheduler_capabilities",
                "attention_implementations", "module_classes", "packages", "optional_kernel_bindings",
                "generation_defaults", "model_dtype", "model_device", "runtime_precision", "limits",
            ) if key in runtime}
            profile["runtime_retrieval"] = (
                "Full requested settings, compatibility explanations and limitations are in the bound observation artifact.")
            steps.append("runtime_details_retrievable")
    if count() > static_target:
        profile = value.get("performance_observation") or {}
        matrix = profile.get("profile_matrix") or {}
        entries = matrix.get("entries")
        if isinstance(entries, list):
            matrix["entries"] = [{key: row.get(key) for key in (
                "profile_id", "prompt_index", "prompt_characters", "profile_max_new_tokens",
                "evidence_path", "normal_request", "hotspots",
            ) if key in row} for row in entries]
            matrix["full_entry_evidence"] = (
                "Read each evidence_path for measurements, phases and ranked operators.")
            steps.append("profile_matrix_measurements_retrievable")
    if count() > static_target:
        profile = value.get("performance_observation") or {}
        matrix = profile.get("profile_matrix") or {}
        entries = matrix.get("entries")
        if isinstance(entries, list):
            matrix["entries"] = [{key: row.get(key) for key in (
                "profile_id", "prompt_index", "evidence_path") if key in row}
                                 for row in entries]
            matrix["summary"] = "Matrix details are retrievable from the listed evidence paths."
            steps.append("profile_matrix_index_only")
    if count() > static_target:
        # These values drive host orchestration and remain in context.json; they
        # do not help the model choose or justify a source change.
        removed = []
        for key in ("config_hash", "workspace", "pending_analyses", "history_total",
                    "memory_index", "budget", "agent", "model", "evidence_root", "engine_url"):
            if key in value:
                value.pop(key)
                removed.append(key)
        if removed:
            value["control_metadata_retrieval"] = (
                "Host-only paths, hashes and budgets remain in the frozen context.json and campaign state.")
            steps.append("control_metadata_retrievable")
        profile = value.get("performance_observation") or {}
        removed = []
        for key in ("schema_version", "kind", "profiler", "profiler_scopes", "precision", "limitations",
                    "source_hash", "profile_hash", "observed_at", "context_summary"):
            if key in profile:
                profile.pop(key)
                removed.append(key)
        if removed:
            profile["profile_metadata_retrieval"] = (
                "Profiler scope, hashes and limitations remain in the bound observation artifact.")
            steps.append("profile_metadata_retrievable")
    if count() > static_target:
        profile = value.get("performance_observation") or {}
        rows = profile.get("top_cpu_ops")
        if isinstance(rows, list) and len(rows) > 3:
            profile["top_cpu_ops"] = rows[:3]
            profile["operator_inventory"] = "Additional ranked operators remain in the bound observation artifact."
            steps.append("operator_tail_retrievable")
    if count() > static_target:
        # Compaction annotations are navigation hints, not evidence. Keep the
        # pointers while avoiding a collection of prose consuming tool space.
        retrieval_keys = [key for key in value if key.endswith("_retrieval")]
        for key in retrieval_keys:
            value[key] = "See frozen context.json or the bound evidence artifact."
        profile = value.get("performance_observation") or {}
        for key in [key for key in profile if key.endswith("_retrieval")]:
            profile[key] = "See the bound observation artifact."
        if retrieval_keys or profile:
            steps.append("retrieval_hints_compacted")
    if count() > static_target and "agent_interface" in value:
        # The live tool schemas are authoritative and already name every
        # available capability, so this descriptor is optional under pressure.
        value.pop("agent_interface")
        steps.append("capability_descriptor_redundant")
    if count() > static_target:
        profile = value.get("performance_observation") or {}
        matrix = profile.get("profile_matrix") or {}
        entries = matrix.get("entries")
        if isinstance(entries, list):
            profile["profile_matrix"] = {
                "coverage": matrix.get("coverage"),
                "selected_cases": matrix.get("selected_cases"),
                "entries": entries,
            }
            steps.append("profile_matrix_annotations_retrievable")
    if count() > static_target:
        profile = value.get("performance_observation") or {}
        changed = False
        for key in ("top_cpu_ops", "top_device_ops", "top_device_operator_ops"):
            rows = profile.get(key)
            if isinstance(rows, list):
                profile[key] = [{field: row.get(field) for field in (
                    "name", "calls", "self_cpu_seconds", "self_device_seconds")
                                 if field in row} for row in rows[:3]]
                changed = True
        if changed:
            steps.append("operator_rows_compacted")
    if count() > static_target and isinstance(value.get("protocol"), dict):
        protocol = value["protocol"]
        value["protocol"] = {key: protocol[key] for key in (
            "objective", "min_gain", "confidence", "max_profile_regression",
            "correctness_mode", "exact_tokens", "limits",
        ) if key in protocol}
        value["protocol"]["full_protocol"] = "See context.json."
        steps.append("evaluation_protocol_compacted")
    if count() > static_target:
        # Preserve every decision-relevant value introduced by the deeper
        # modification contract while removing duplicate paths and prose. The
        # unabridged values remain in the frozen context for evidence reads.
        surfaces = value.get("modification_surfaces") or {}
        allowed = surfaces.get("allowed", [])
        scheduler = surfaces.get("scheduler") or {}
        if surfaces:
            value["modification_surfaces"] = {
                "allowed": allowed,
                "runtime_config_fields": surfaces.get("runtime_config", []),
                "request_scheduler_values": scheduler.get("values", []),
                "execution_adapter_authorized": "execution_adapter" in allowed,
                "native_kernel_authorized": "native_kernel" in allowed,
                "admission_gates": ["source", "build", "runtime", "correctness", "paired_benchmark"],
            }
        adapter = value.get("engine_adapter") or {}
        if adapter:
            value["engine_adapter"] = {key: adapter[key] for key in (
                "adapter", "model_format", "model_loader", "required_packages",
                "build_command", "build_timeout_seconds",
            ) if key in adapter}
        loop = value.get("loop_protocol") or {}
        if loop:
            value["loop_protocol"] = {
                "telemetry": {"enabled": (loop.get("telemetry") or {}).get("enabled", False)},
                "profiling": {key: (loop.get("profiling") or {}).get(key) for key in (
                    "enabled", "coverage", "max_profiles_per_session")},
                "probe": {"enabled": (loop.get("probe") or {}).get("enabled", False)},
                "refinement": {key: (loop.get("refinement") or {}).get(key) for key in (
                    "enabled", "max_revisions")},
            }
        environment = value.get("environment") or {}
        inference = environment.get("inference") or {}
        if environment and inference:
            value["environment"] = {
                "inference_python": inference.get("python"),
                "packages": inference.get("packages", {}),
                "package_count": inference.get("package_count"),
                "runtime": inference.get("runtime"),
                "full_environment": "See context.json.",
            }
        steps.append("execution_contract_compacted")
    if count() > static_target:
        removed = []
        for key in ("telemetry_session", "workload_roles", "instructions"):
            if key in value:
                value.pop(key)
                removed.append(key)
        if removed:
            steps.append("redundant_session_metadata_removed")
    if count() > static_target:
        workload = value.get("workload") or {}
        changed = False
        for profile in [workload, *workload.get("profiles", [])]:
            prompts = profile.get("prompts")
            if isinstance(prompts, list):
                descriptors = [{key: prompt[key] for key in (
                    "index", "characters", "content_in_frozen_host_context") if key in prompt}
                    for prompt in prompts if isinstance(prompt, dict)]
                profile.pop("prompts")
                profile["prompt_count"] = len(prompts)
                if descriptors:
                    profile["prompt_descriptors"] = descriptors
                changed = True
        if changed:
            workload["prompt_retrieval"] = "Full frozen prompts remain in context.json."
            steps.append("workload_prompt_previews_retrievable")
    if count() > static_target:
        # This is the final semantic compaction tier. Keep the values required
        # to select a bottleneck and bind a proposal, while leaving verbose
        # runtime inventories and request annotations in the evidence artifact.
        profile = value.get("performance_observation") or {}
        normal = profile.get("normal_request") or {}
        if normal:
            profile["normal_request"] = {key: normal[key] for key in (
                "backend_wall_seconds", "queue_seconds", "input_tokens", "output_tokens",
                "output_tokens_per_second", "ttft_seconds", "tpot_seconds",
            ) if key in normal}
        runtime = profile.get("runtime") or {}
        if runtime:
            profile["runtime"] = {key: runtime[key] for key in (
                "execution_path", "request_scheduler", "continuous_batching",
                "batching_mode", "scheduler_capabilities",
                "attention_implementations", "packages", "generation_defaults",
                "model_dtype", "model_device", "runtime_precision", "limits",
            ) if key in runtime}
        for key in list(profile):
            if key.endswith("_retrieval") or key in ("measurement_inventory", "operator_inventory"):
                profile.pop(key)
        profile["full_profile"] = "See the bound observation artifact."
        steps.append("profile_decision_summary")
    if count() > static_target:
        # The brief intentionally duplicates evidence for first-screen access.
        # Under a small context window retain decision signals, not inventories.
        brief = value.get("optimization_brief") or {}
        if brief:
            model = brief.get("model_summary") or {}
            runtime = brief.get("runtime_facts") or {}
            brief["model_summary"] = {key: model[key] for key in (
                "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                "layer_type_counts", "head_dim") if key in model}
            brief["runtime_facts"] = {key: runtime[key] for key in (
                "execution_path", "request_scheduler", "continuous_batching", "batching_mode",
                "attention_implementations", "precision", "device") if key in runtime}
            brief["workload_profiles"] = (brief.get("workload_profiles") or [])[:6]
            brief["top_operator_families"] = (brief.get("top_operator_families") or [])[:3]
            brief["relevant_failures"] = (brief.get("relevant_failures") or [])[-3:]
            brief.pop("profiler_diagnostics", None)
            brief.pop("evidence_paths", None)
            brief.pop("details", None)
            steps.append("optimization_brief_decision_summary")
    if count() > static_target and isinstance(value.get("environment"), dict):
        environment = value["environment"]
        packages = environment.get("packages") or {}
        environment["packages"] = {key: packages[key] for key in (
            "torch", "transformers", "bitsandbytes", "triton", "vllm") if key in packages}
        environment.pop("runtime", None)
        steps.append("environment_package_inventory_retrievable")
    if count() > static_target and not analysis_only and value.get("optimization_brief"):
        value = {key: value[key] for key in (
            "campaign_id", "iteration", "stable_ref", "requirements", "optimization_brief",
            "candidate_contract", "transfer_hints", "candidate_feedback", "refinement", "refinement_target",
            "candidate_engine_config",
            "knowledge",
            "instructions", "agent_interface", "source_retrieval", "previous", "previous_analysis",
        ) if key in value}
        value["evidence_retrieval"] = "Read the bound observation and history files only for missing facts. Read engine.py before editing."
        steps.append("planning_brief_only")
    return value, steps, count()


class _EngineTokenCounter:
    """Use the active model tokenizer when available, with the old estimate as fallback."""

    def __init__(self, url, timeout):
        from qwen_agent.utils.tokenization_qwen import tokenizer
        self.fallback = tokenizer
        self.client = EngineClient(url, min(timeout, 30))
        self.max_context = None
        try:
            health = self.client.health()
            self.exact = "tokenize" in health.get("capabilities", [])
            self.max_context = health.get("max_context")
        except Exception:
            self.exact = False

    def count_tokens(self, text):
        if self.exact:
            return self.client.tokenize(text=text)["count"]
        return self.fallback.count_tokens(text)

    def count_chat(self, system, context, tools):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
        if self.exact:
            return self.client.tokenize(messages=messages, tools=tools,
                                        add_generation_prompt=True)["count"]
        return self.fallback.count_tokens(system) + self.fallback.count_tokens(messages[1]["content"])


def _openai_tool_schemas(tools):
    return [{"type": "function", "function": {
        "name": tool.name, "description": tool.description, "parameters": tool.parameters,
    }} for tool in tools]


def _run_qwen_agent(url, context, workspace, trace, prompt, api_key, analysis_only=False, smoke=False):
    from qwen_agent.agents import Assistant
    from qwen_agent.llm.base import ModelServiceError

    settings = context.get("agent", {})
    session_id = uuid.uuid4().hex
    submission = {}
    assistant_inputs = {}
    input_artifact = Path(trace).with_name("agent-input.json")
    audited_calls = []
    system = prompt
    previous = context.get("previous") or {}
    if previous.get("iteration") is not None and not smoke:
        system += (
            f"\nAnalysis target: iteration {previous['iteration']} in context.previous. "
            f"Begin previous_analysis with 'Iteration {previous['iteration']}:'. "
            "There may be an older analysis backlog; this target can differ from the latest experiment. "
            "Analyze its recorded decision and hypothesis. Do not substitute an analysis of a later iteration."
        )
    if not analysis_only and not smoke:
        system += (
            "\nEvidence policy for this session supersedes any earlier restriction on evidence reads: "
            "you may use read_evidence and list_evidence to inspect existing runtime logs, decisions, "
                "change patches and benchmark summaries. Read only relevant sections and keep tool calls concise. "
            "The source in context is raw Python; replacement spans must use its actual quotes and newlines. "
            "A backend in the arguments dictionary passed via **arguments to from_pretrained is applied "
            "during model loading exactly like a direct keyword; moving it is not a different experiment."
        )
        if context.get("telemetry_session"):
            system += (
                "\nBefore choosing a bottleneck, call read_telemetry after at least one ordinary model "
                "response. Use it to distinguish queue, prefill, decode, tokenization and host overhead. "
                "This live Agent workload is diagnostic evidence only and never an admission benchmark."
            )
        system += (
            "\noptimization_brief and knowledge are fields of the supplied JSON, not workspace filenames. "
            "Check the supplied knowledge recipes and their validation scope before spending tool calls on exploration. "
            "A component-validated recipe is a hypothesis to test, not proof of whole-model improvement. "
            "\nUse read_implementation to inspect the loaded model forward and hotspot helpers, not just the engine wrapper. "
            "Pure tensor fallback helpers may be candidates for local fusion or compilation when torch/triton are available; "
            "preserve dtype, state and all fallbacks, and verify outputs. "
            "Use read_telemetry once and inspect only objective-relevant matrix/source evidence. Before "
            "report_blocked, assess every modification_surfaces.allowed value. Missing flash-attn rules "
            "out only flash attention, not NF4 GEMM, scheduling, adapters, cache policy, or an authorized "
            "native kernel with a frozen build_command. Name each plausible surface's concrete blocker; "
            "do not invent gains or unsafe edits."
        )
        no_change_streak = 0
        for item in reversed(context.get("history", [])):
            decision = item.get("decision") or {}
            if not decision:
                continue
            if decision.get("reason") not in {"no_change", "duplicate_change"}:
                break
            no_change_streak += 1
        if no_change_streak >= 3:
            system += (
                f"\nSearch stalled: {no_change_streak} consecutive iterations produced no new candidate. "
                "Prior attempts did not exhaust the optimization space. Before claiming no viable change, "
                "read a startup/runtime log through read_evidence, inspect model_features and the captured "
                "environment, and look for an untested bottleneck outside the previously repeated attention "
                "backend and TokenClock edits. Consider model loading/runtime configuration, host overhead, "
                "generation execution, or model-specific kernels already available in this environment. "
                "Use the evidence to select one small, distinct source change. If blocked, call report_blocked "
                "with the new direction examined and its specific blocker; do not repeat an exhaustion claim. "
                "Never change the frozen workload, correctness gates, precision or measurement standards."
            )
    if not analysis_only and previous.get("analysis_pending"):
        system += (
            "\nThe prior iteration still needs analysis. Treat `previous_analysis` as a placeholder if it says "
            "analysis is pending. Before selecting a hypothesis, analyze the recorded previous decision and "
            "hypothesis in `previous` and `history`; state measured gain and confidence bounds where available. "
            "Submit that real analysis in `previous_analysis`; never copy the placeholder. Choose a materially "
            "different configuration or implementation after an inconclusive result; do not treat it as a proven regression."
        )
    packages = context.get("environment", {}).get("inference", {}).get("packages", {})
    installed = {name.casefold().replace("_", "-") for name in packages}
    model_type = context.get("model_features", {}).get("model_type", "")
    supported = QWEN35_ATTENTION_BACKENDS if model_type.startswith("qwen3_5") else TRANSFORMERS_ATTENTION_BACKENDS
    transformers_version = packages.get("transformers", "the captured Transformers release")
    system += (f"\nValid `attn_implementation` values for this {model_type or 'Transformers'} model with "
               f"Transformers {transformers_version} are: {', '.join(sorted(supported))}. "
               "A package name is not automatically a valid backend identifier; xformers is not an "
               "`attn_implementation` value for this model.")
    system += ("\nValid literal `cache_implementation` values for the captured Transformers runtime are: "
               f"{', '.join(sorted(TRANSFORMERS_CACHE_IMPLEMENTATIONS))}. `sdpa` is an attention "
               "implementation, not a cache implementation; there is no `SdpaCache` in this runtime.")
    rejected_families = sorted(rejected_attention_backend_families(context.get("history", [])))
    if rejected_families:
        system += ("\nPrior rejected candidates used these attention backend families: "
                   f"{', '.join(rejected_families)}. Read the associated decisions and change_patch before "
                   "selecting a hypothesis. Avoid repeating failed changes. A valid new experiment must "
                   "identify a concrete difference from the previous source change.")
    if context.get("history"):
        system += ("\nRead change.patch under each relevant history evidence_path; compare actual edits, not just hypothesis text. "
                   "Read older patches through read_evidence when needed. Similar wording does not establish "
                    "a different implementation. Choose a different mechanism after repeatedly negative results.")
    if not analysis_only and context.get("archived_candidate_results"):
        system += ("\nArchived candidate results were measured under this same parent, frozen configuration and profile. "
                   "Read their decision.json and change.patch through each evidence_path before choosing a change. "
                   "The host rejects exact semantic duplicates across these campaigns. A new hypothesis label "
                   "does not make an already tested implementation new. Archived observations cannot replace "
                   "the current campaign's fresh bottleneck evidence.")
    for backend, package in OPTIONAL_BACKEND_PACKAGES.items():
        if package not in installed:
            system += (f"\nThe captured inference environment does not contain `{package}`. Do not select "
                       f"`{backend}` in `engine.py`; use an installed backend or report a blocked proposal.")
    if analysis_only:
        target = previous.get("iteration")
        system = (
            "You review a recorded experiment. This is an analysis-only session, not a new optimization. "
            f"Analysis target: iteration {target} in context.previous. "
            "Use context.target_evidence_directory with list_evidence and context.decision_evidence_path "
            "with read_evidence; paths are evidence-root-relative. Do not guess directory names. "
            "Read its decision and, when present, correctness, verification and benchmark reports. "
            "When context.correctness_summary is present, use it instead of paging through raw token arrays; "
            "cite its evidence_path and distinguish generation/repeat/isolation failures from numerical failures. "
            "Never claim an RMS, logit or numerical threshold failure when every summarized numerical check passed. "
            "Do not select a new hypothesis, propose an implementation, edit files or analyze the current stable engine. "
            "Finish by CALLING submit_candidate with a brief hypothesis label identifying this review, "
            f"previous_analysis beginning 'Iteration {target}:' and reflection "
            "{iteration, hypothesis_status: supported/refuted/inconclusive, evidence_paths, explanation, next_action}. "
            "Cite existing evidence from the assigned iteration. No experiment_plan is needed. "
            "Infrastructure failures, interruptions and absent measurements are inconclusive; never invent a gain. "
            "Distinguish measured acceptance, predicted magnitude and causal interpretation. "
            "Use review_measurement_status: observation.json describes the stable parent before edits. "
            "Both normal_request and timing in observation.json belong to the same stable parent; timing is the profiler run. "
            "When formal_comparison_available is false, do not label any of those numbers as candidate metrics, "
            "and do not claim a numerical candidate speedup or regression. "
            "When formal_comparison_available is true, do not claim that benchmarks were absent or incomplete; "
            "cite verification.json or the benchmark summary. A profile protection decision may have null gain even "
            "though experiment_verification contains a completed paired comparison. "
            "Keep previous_analysis under 1800 characters, explanation under 1000 and next_action under 600. "
            "A future action may describe an investigation, but do not design another candidate in this review. "
            "Submit once and stop; prose without the tool call does not complete the review."
        )
    if smoke:
        system = "Call submit_candidate exactly once with a short readiness hypothesis. Do not use other tools."
    model = context.get("model", {}).get("name") or "vibeforge-local"
    def make_assistant(system_message, function_list, max_tokens, tool_choice=None):
        thinking = settings.get("thinking") or {"enabled": False}
        generate_cfg = {
            "use_raw_api": True,
            "max_tokens": max_tokens,
            "max_input_tokens": _agent_input_budget(context, max_tokens),
            "request_timeout": settings.get("request_timeout", 180),
            "temperature": thinking.get("temperature", 0.6) if thinking.get("enabled") else 0,
            "extra_body": {"chat_template_kwargs": {
                "enable_thinking": bool(thinking.get("enabled", False)),
            }},
        }
        if thinking.get("enabled"):
            generate_cfg.update(top_p=thinking.get("top_p", 0.95),
                                top_k=thinking.get("top_k", 20))
            generate_cfg["extra_body"]["min_p"] = thinking.get("min_p", 0.0)
        if tool_choice:
            generate_cfg["tool_choice"] = tool_choice
        assistant = Assistant(
            llm={
            "model": model,
            "model_type": "oai",
            "model_server": url.rstrip("/") + "/v1",
            "api_key": api_key or "EMPTY",
            "generate_cfg": generate_cfg,
            },
            function_list=function_list,
            system_message=system_message,
        )
        assistant_inputs[id(assistant)] = {
            "system_prompt": system_message,
            "tools": [{"name": tool.name, "description": tool.description,
                       "parameters": tool.parameters} for tool in function_list],
            "generation": {key: value for key, value in generate_cfg.items()},
            "model": model,
        }
        return assistant

    def consume(active_assistant, active_messages, event_name, deadline, tool_call_limit=None):
        tool_call_limit = settings.get("max_turns", 6) if tool_call_limit is None else tool_call_limit
        record = {**assistant_inputs[id(active_assistant)], "event": event_name,
                  "messages": copy.deepcopy(active_messages)}
        record["sha256"] = digest(record)
        audited_calls.append(record)
        write_json(input_artifact, {"schema_version": 1, "calls": audited_calls})
        with closing(active_assistant.run(messages=active_messages)) as updates:
            for state in updates:
                snapshot = [item.model_dump() if hasattr(item, "model_dump") else item for item in state]
                append_record(trace, {"session_id": session_id, "event": event_name, "messages": snapshot})
                latest_answer["messages"] = snapshot
                answer = "\n".join(item.get("content", "") for item in snapshot
                                    if item.get("role") == "assistant" and item.get("content"))
                if answer:
                    latest_answer["text"] = answer
                if time.monotonic() >= deadline:
                    raise TimeoutError("Qwen-Agent session budget exhausted")
                if submission.get("complete"):
                    break
                if (context.get("candidate_contract") and not analysis_only
                        and attempt_state.get("rejected_submission_count", 0) >= 2):
                    append_record(trace, {
                        "event": "qwen_agent_rejected_submission_limit_reached",
                        "session_id": session_id,
                        "rejected_submissions": attempt_state["rejected_submission_count"],
                    })
                    break
                completed_calls = latest_answer["prior_calls"] + sum(item.get("role") == "function" for item in snapshot)
                if completed_calls >= tool_call_limit:
                    raise RuntimeError("Qwen-Agent reached the tool-call budget without a valid submission")

    attempt_state = {}
    tools = _assistant_tools(workspace, trace, context, submission, analysis_only, smoke,
                             attempt_state=attempt_state)
    tokenizer = _EngineTokenCounter(url, settings.get("request_timeout", 180))
    tool_schemas = _openai_tool_schemas(tools)
    if not context.get("model", {}).get("max_context") and tokenizer.max_context:
        context = copy.deepcopy(context)
        context.setdefault("model", {})["max_context"] = tokenizer.max_context
    input_budget = _agent_input_budget(context, settings.get("max_new_tokens", 512))
    # Reserve room for tool results without starving the initial diagnostic
    # brief on smaller local context windows. The engine still enforces the
    # hard input limit on every subsequent turn.
    history_reserve = 1024
    fixed_tokens = tokenizer.count_chat(system, {}, tool_schemas)
    planning_ratio_target = int(input_budget * 0.55)
    static_target = (input_budget - history_reserve if analysis_only or smoke else
                     min(input_budget - history_reserve, max(fixed_tokens + 2048, planning_ratio_target)))
    model_context, compaction_steps, estimated_tokens = _fit_model_context(
        context, system, input_budget, history_reserve, tokenizer, analysis_only=analysis_only,
        count_context=lambda value: tokenizer.count_chat(system, value, tool_schemas),
        max_static_tokens=static_target if not analysis_only and not smoke else None)
    serialized_context = json.dumps(model_context, ensure_ascii=False)
    append_record(trace, {"event": "agent_context_budget", "session_id": session_id,
        "estimated_static_tokens": estimated_tokens, "max_input_tokens": input_budget,
        "fixed_prompt_and_tool_tokens": fixed_tokens,
        "static_target_tokens": static_target,
        "planning_ratio_target_tokens": planning_ratio_target,
        "compaction_steps": compaction_steps,
        "tool_history_reserve": history_reserve, "remaining_tokens": input_budget - estimated_tokens,
        "scope": ("active engine tokenizer with chat template and tool schemas"
                  if tokenizer.exact else "Qwen-Agent fallback estimate; engine independently checks actual length"),
        "exact": tokenizer.exact})
    if estimated_tokens > input_budget - history_reserve:
        raise ValueError("Agent static context leaves insufficient tool-history space within the model context limit")
    messages = [{"role": "user", "content": serialized_context}]
    deadline = time.monotonic() + settings.get("timeout_seconds", 180)
    latest_answer = {"text": "", "messages": [], "prior_calls": 0}
    active_session = session_id
    recovery_error = None
    emergency_recovery = False
    max_turns = settings.get("max_turns", 6)
    planning_limit = max_turns
    if context.get("candidate_contract") and not analysis_only and not smoke and max_turns > 1:
        # Reserve implementation, syntax checking and submission, not just a
        # terminal action after every available call was spent reading files.
        planning_limit -= 4 if max_turns >= 8 else 1

    def compact_action_context(instruction):
        parent = Path(context["candidate_contract"].get("parent_source", ""))
        stable_path = parent / "engine.py"
        candidate_path = Path(workspace) / "engine.py"
        stable_engine = stable_path.read_text()[:12000] if stable_path.is_file() else ""
        candidate_engine = candidate_path.read_text()[:12000] if candidate_path.is_file() else ""
        edits = candidate_edit_summary(parent, workspace)
        previous_row = context.get("previous") or {}
        previous = {key: previous_row[key] for key in (
            "iteration", "candidate", "hypothesis", "decision", "analysis_pending",
            "interpretation", "reflection",
        ) if key in previous_row}
        compact_profile = _model_context({
            "performance_observation": context.get("performance_observation", {})
        }).get("performance_observation", {})
        history = [{key: row[key] for key in (
            "iteration", "candidate", "hypothesis", "decision", "evidence_path",
        ) if key in row} for row in context.get("history", [])]
        return {
            "iteration": context.get("iteration"),
            "stable_ref": context.get("stable_ref"),
            "profile_hash": context.get("profile_hash"),
            "requirements": context.get("requirements"),
            "precision": context.get("precision"),
            "previous": previous,
            "previous_analysis": context.get("previous_analysis"),
            "history": history,
            "knowledge": context.get("knowledge", []),
            "archived_candidate_results": context.get("archived_candidate_results", []),
            "candidate_contract": {
                "version": context["candidate_contract"].get("version", 1),
                "observation_paths": context["candidate_contract"].get("observation_paths", []),
                "history_refs": context["candidate_contract"].get("history_refs", []),
            },
            "performance_observation": compact_profile,
            "stable_engine": stable_engine[:6000],
            "candidate_engine": candidate_engine[:6000],
            **edits,
            "last_response": latest_answer["text"][-2400:],
            "last_rejected_submission": attempt_state.get("last_rejected_arguments"),
            "last_gate_result": attempt_state.get("last_rejection"),
            "instruction": instruction,
        }

    finalization_attempted = False

    def finalize_candidate(reason):
        nonlocal finalization_attempted
        finalization_attempted = True
        final_context = compact_action_context(
            "Finish with exactly one structured action. If workspace_changed is true, submit the existing "
            "change with a complete experiment_plan grounded in the supplied measurements. If it is false, "
            "report_blocked with a concrete evidence-based reason. Do not read, edit or answer in prose."
        )
        action = "submit_candidate" if final_context["workspace_changed"] else "report_blocked"
        final_tools = [tool for tool in _assistant_tools(
            workspace, trace, context, submission, attempt_state=attempt_state
        ) if tool.name == action]
        append_record(trace, {"event": "qwen_agent_planning_finalization_started",
                              "session_id": session_id, "remaining_tool_calls": 1,
                              "remaining_seconds": max(0, deadline - time.monotonic()),
                              "reason": reason, "action": action})
        try:
            consume(make_assistant(
                system + f"\nFinalize now by calling {action} exactly once; do no more exploration.",
                final_tools, min(settings.get("max_new_tokens", 512), 1024),
                tool_choice={"type": "function", "function": {"name": action}}),
                [{"role": "user", "content": json.dumps(final_context, ensure_ascii=False)}],
                "qwen_agent_planning_finalization", deadline)
        except RuntimeError as error:
            if submission.get("complete") or "tool-call budget" not in str(error):
                raise
        except ModelServiceError as error:
            if not _recoverable_model_error(error):
                raise
            return str(error)
        return None
    try:
        consume(make_assistant(system, tools, settings.get("max_new_tokens", 512)),
                messages, "qwen_agent_update", deadline, planning_limit)
    except RuntimeError as error:
        if submission.get("complete") or "tool-call budget" not in str(error):
            raise
    except ModelServiceError as error:
        message = str(error)
        if (submission.get("complete") or smoke
                or not _recoverable_model_error(error)):
            raise
        recovery_error = message
    recovered = False
    if (not submission.get("complete") and not analysis_only and not smoke
            and context.get("candidate_contract") and not recovery_error):
        prior_calls = sum(item.get("role") == "function" for item in latest_answer["messages"])
        remaining_calls = max_turns - prior_calls
        if remaining_calls > 0 and time.monotonic() < deadline:
            # Assistant.run can finish with prose while the experiment task is
            # unfinished. Give the model one explicit action feedback turn,
            # preserving its evidence/tool history and the original hard budgets.
            recovered = True
            continuation = (
                "Your previous response ended without completing the experiment task. "
                "Continue by calling tools now: implement one evidence-justified change and submit its experiment_plan, "
                "or call report_blocked with a concrete reason if no justified change is available. "
                "Do not continue open-ended discussion. No-change cannot pass the candidate gate. "
                f"This is the only continuation; {remaining_calls} tool calls remain in the original session budget."
            )
            latest_answer["prior_calls"] = prior_calls
            rejected_submission_limit = attempt_state.get("rejected_submission_count", 0) >= 2
            if remaining_calls == 1 or rejected_submission_limit:
                recovery_error = finalize_candidate(
                    "rejected_submission_limit" if rejected_submission_limit else "reserved_tool_call")
                continuation_messages = None
            else:
                continuation_context = compact_action_context(continuation)
                continuation_messages = [
                    {"role": "user", "content": json.dumps(continuation_context, ensure_ascii=False)},
                    {"role": "assistant", "content": latest_answer["text"][-2400:]},
                    {"role": "user", "content": continuation},
                ]
                append_record(trace, {"event": "qwen_agent_planning_continuation_started", "session_id": session_id,
                                      "remaining_tool_calls": remaining_calls,
                                      "remaining_seconds": max(0, deadline - time.monotonic())})
            try:
                if continuation_messages is not None:
                    action_tools = [tool for tool in tools if tool.name in {
                        "replace_text", "write_file", "check_python", "submit_candidate", "report_blocked"}]
                    consume(make_assistant(system + "\n" + continuation +
                            " Exploration has ended. Use the supplied source and evidence to implement and submit, "
                            "or explain the specific missing prerequisite; no further reading tools are available.",
                            action_tools, settings.get("max_new_tokens", 512)),
                            continuation_messages, "qwen_agent_planning_continuation", deadline, max_turns - 1)
            except RuntimeError as error:
                if submission.get("complete") or "tool-call budget" not in str(error):
                    raise
            except ModelServiceError as error:
                if not _recoverable_model_error(error):
                    raise
                recovery_error = str(error)
            if (not submission.get("complete") and not recovery_error and not finalization_attempted
                    and time.monotonic() < deadline):
                recovery_error = finalize_candidate("continuation_incomplete")
    if not submission.get("complete") and analysis_only and not smoke:
        # One bounded model-authored finalization; attribution/evidence checks
        # remain in the same submit tool and no host prose can bypass them.
        recovered = True
        active_session = uuid.uuid4().hex
        final_tools = _assistant_tools(workspace, trace, context, submission, analysis_only=True, submit_only=True,
                                       attempt_state=attempt_state)
        final_context = _model_context(context, analysis_only=True)
        final_context["last_response"] = latest_answer["text"][:1600]
        final_context["protocol_error"] = recovery_error
        if time.monotonic() >= deadline:
            raise TimeoutError("Qwen-Agent analysis session budget exhausted before finalization")
        consume(make_assistant(system + " Finalize the assigned review now with submit_candidate; no more evidence reads.",
                               final_tools, settings.get("max_new_tokens", 512),
                               tool_choice={"type": "function", "function": {"name": "submit_candidate"}}),
                [{"role": "user", "content": json.dumps(final_context, ensure_ascii=False)}],
                "qwen_agent_analysis_finalization", deadline)
    if not submission.get("complete") and not analysis_only and not smoke:
        if context.get("candidate_contract"):
            # A protocol failure is not a candidate, and fabricated recovery prose is not analysis.
            return {"session_id": active_session, "mode": "qwen_agent", "complete": True,
                    "hypothesis": "Agent did not complete a justified candidate submission.",
                    "blocked": True, "blocked_reason": "agent_incomplete_submission",
                    "submission_origin": "controller_recovery", "recovery_error": recovery_error}
        recovered = True
        active_session = uuid.uuid4().hex
        recovery_context = {
            "iteration": context.get("iteration"),
            "objective": context.get("requirements", {}).get("objective"),
            "previous": context.get("previous"),
            "last_response": latest_answer["text"][:2400],
            "recovery_error": recovery_error,
            "stable_engine": context.get("source", "")[:12000] if recovery_error else "",
            "candidate_engine": ((Path(workspace) / "engine.py").read_text()[:12000]
                                 if recovery_error and (Path(workspace) / "engine.py").is_file() else ""),
        }
        recovery_system = (
            "Recovery step: call submit_candidate exactly once now. Do not edit files or answer in prose. "
            "Use a concise hypothesis and summary based on the supplied response and current candidate. "
            "If no change was made, say so in summary. Include previous_analysis, or state that this is the initial iteration."
        )
        if recovery_error:
            recovery_system += (
                (" The model server rejected the previous response as a malformed or truncated tool call. "
                 if "malformed" in recovery_error.casefold() else
                 " The accumulated conversation exceeded the engine context limit; this is a compact recovery. ") +
                "Compare stable_engine with candidate_engine and accurately describe the existing change; "
                "send one valid structured submit_candidate call."
            )
        recovery_tools = _assistant_tools(workspace, trace, context, submission, submit_only=True,
                                          attempt_state=attempt_state)
        recovery_assistant = make_assistant(
            recovery_system,
            recovery_tools,
            settings.get("max_new_tokens", 512),
            {"type": "function", "function": {"name": "submit_candidate"}},
        )
        recovery_messages = [{"role": "user", "content": json.dumps(recovery_context, ensure_ascii=False)}]
        try:
            consume(recovery_assistant, recovery_messages, "qwen_agent_recovery_update", deadline)
        except ModelServiceError as error:
            if not _recoverable_model_error(error):
                raise
            emergency_recovery = True
            recovery_error = str(error)
            fallback_arguments = {
                "hypothesis": "No valid optimization submission was recovered after two model protocol/context errors.",
                "previous_analysis": _previous_analysis_for_recovery(context),
                "summary": ("The local model rejected both the main session and compact recovery. The candidate workspace is "
                           "retained for diagnosis; skip candidate evaluation and start a fresh iteration."),
            }
            submit_tool = next(tool for tool in recovery_tools if tool.name == "submit_candidate")
            response = json.loads(submit_tool.call(json.dumps(fallback_arguments, ensure_ascii=False)))
            append_record(trace, {"event": "qwen_agent_emergency_submission",
                                  "arguments": fallback_arguments, "result": response,
                                  "error": str(error)})
            if not response.get("submitted"):
                raise RuntimeError("Emergency submission was rejected: " + response.get("error", "unknown reason"))
    if not submission.get("complete"):
        raise RuntimeError("Qwen-Agent submission recovery finished without calling submit_candidate")
    return {**submission, "session_id": active_session, "mode": "qwen_agent", "complete": True,
            "recovered_submission": recovered,
            "submission_origin": "controller_recovery" if emergency_recovery else "agent",
            "recovery_error": recovery_error}


def smoke_agent(client, max_new_tokens=80, trace=None):
    with tempfile.TemporaryDirectory(prefix="vibeforge-qwen-agent-smoke-") as directory:
        root = Path(directory)
        result = _run_qwen_agent(
            client.url,
            {"agent": {"max_new_tokens": max_new_tokens, "request_timeout": client.timeout}},
            root,
            trace or root / "trace.jsonl",
            "",
            "vibeforge-smoke",
            smoke=True,
        )
    return {"passed": True, "framework": "Qwen-Agent", "submission": result}


def run_agent(url, context, workspace, trace, prompt_path, api_key="", analysis_only=False):
    if context.get("agent", {}).get("mode") == "scripted":
        action_list = context["agent"].get("actions", [])
        action = action_list[min(context["iteration"] - 1, len(action_list) - 1)] if action_list else {}
        if context.get("refinement"):
            action = action.get("revision", action)
        if not analysis_only:
            tools = AgentTools(workspace, trace)
            for edit in action.get("edits", []):
                tools.execute(edit["tool"], edit["arguments"])
        result = {
            "session_id": uuid.uuid4().hex,
            "mode": "scripted_test_fixture",
            "hypothesis": action.get("hypothesis", "No code change; test fixture iteration."),
            "previous_analysis": "The fixture records a predetermined decision; causal optimization ability is not tested.",
            "summary": "Scripted integration fixture completed.",
            "complete": True,
        }
        if context.get("candidate_contract"):
            previous = context.get("previous")
            reflection_evidence = ([f"iterations/{previous['iteration']:04d}/decision.json"]
                                   if previous else [])
            if previous and (previous.get("decision") or {}).get("reason") == "correctness_failed":
                correctness = f"iterations/{previous['iteration']:04d}/correctness/report.json"
                if (Path(context.get("evidence_root", "")) / correctness).is_file():
                    reflection_evidence.append(correctness)
            result["reflection"] = ({
                "iteration": previous["iteration"], "hypothesis_status": "inconclusive",
                "explanation": "Predetermined fixture; no model causal reasoning is claimed.",
                "next_action": "Run the next fixed integration fixture.",
                "evidence_paths": reflection_evidence,
            } if previous else None)
            result["experiment_plan"] = action.get("experiment_plan", {
                "measured_bottleneck": "Fixture CPU generate/logits hotspot in host profile",
                "cause": "Predetermined redundant arithmetic", "change": action.get("hypothesis", "fixture"),
                "metric": "throughput", "direction": "increase", "expected_change_pct": 1,
                "validation": "Frozen correctness and paired performance evaluation",
                "evidence_paths": context["candidate_contract"]["observation_paths"][:1],
                "history_refs": [previous["iteration"]] if previous else [],
            })
        append_record(trace, result)
        return result

    context = dict(context)
    context["agent"] = dict(context.get("agent", {}))
    framework = context["agent"].get("framework", "qwen-agent")
    context["agent"]["mode"] = "local"
    system = Path(prompt_path).read_text()
    if framework == "codex":
        from vibeforge.agent.codex import run_codex_agent
        return run_codex_agent(url, context, Path(workspace), trace, system, analysis_only=analysis_only)
    if framework != "qwen-agent":
        raise ValueError(f"Unsupported Agent framework: {framework}")
    return _run_qwen_agent(url, context, Path(workspace), trace, system, api_key, analysis_only)


def worker(context_path, workspace, trace, result_path, url, prompt_path, api_key="", analysis_only=False, smoke=False):
    context = read_json(context_path)
    if smoke:
        with tempfile.TemporaryDirectory(prefix="vibeforge-agent-smoke-") as directory:
            framework = context.get("agent", {}).get("framework", "qwen-agent")
            if framework == "codex":
                from vibeforge.agent.codex import run_codex_agent
                submission = run_codex_agent(url, context, Path(directory), trace, "", smoke=True)
                result = {"passed": True, "framework": "Codex", "submission": submission}
            elif framework == "qwen-agent":
                submission = _run_qwen_agent(url, context, Path(directory), trace, "", api_key, smoke=True)
                result = {"passed": True, "framework": "Qwen-Agent", "submission": submission}
            else:
                raise ValueError(f"Unsupported Agent framework: {framework}")
    else:
        result = run_agent(url, context, Path(workspace), trace, prompt_path, api_key, analysis_only)
    write_json(result_path, result)
