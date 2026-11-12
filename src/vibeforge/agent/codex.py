"""Run Codex CLI while keeping inference on the active VibeForge engine."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
import difflib
import re
from collections import Counter
from pathlib import Path
from jsonschema import ValidationError, validate

from vibeforge.agent.backend_policy import (
    QWEN35_ATTENTION_BACKENDS,
    TRANSFORMERS_ATTENTION_BACKENDS,
    TRANSFORMERS_CACHE_IMPLEMENTATIONS,
)
from vibeforge.agent.interface import agent_capabilities
from vibeforge.agent.navigation import candidate_index
from vibeforge.observability.logs import append_record
from vibeforge.engine.client import EngineClient
from vibeforge.storage.atomic import atomic_write, read_json, write_json
from vibeforge.versioning.artifacts import inventory
from vibeforge.versioning.candidate import create_workspace


def _result_schema(analysis_only=False, smoke=False, history_refs=None, bottleneck_metrics=None):
    string = {"type": "string"}
    if smoke:
        return {"type": "object", "properties": {"ready": {"type": "boolean"}, "message": string},
                "required": ["ready", "message"], "additionalProperties": False}
    reflection = {"type": "object", "properties": {
        "iteration": {"type": "integer"}, "hypothesis_status": {
            "type": "string", "enum": ["supported", "refuted", "inconclusive"]},
        "explanation": string, "next_action": string,
        "evidence_paths": {"type": "array", "items": string},
    }, "required": ["iteration", "hypothesis_status", "explanation", "next_action",
                    "evidence_paths"], "additionalProperties": False}
    properties = {
        "hypothesis": string, "previous_analysis": string, "summary": string,
        "blocked": {"type": "boolean"}, "blocked_reason": string,
        "reflection": reflection,
    }
    required = ["hypothesis", "previous_analysis", "summary", "blocked", "blocked_reason", "reflection"]
    if not analysis_only:
        refs = sorted(set(history_refs or []))
        history_items = {"type": "string", "enum": refs} if refs else {"type": "string"}
        properties["experiment_plan"] = {"type": "object", "properties": {
            "measured_bottleneck": string, "cause": string, "change": string,
            "expected_change_pct": {"type": "number", "minimum": 0,
                                    "description": "Positive expected improvement magnitude in percent; never negative for latency reduction. Use 0 only if blocked."}, "validation": string,
            "history_refs": {"type": "array", "items": history_items,
                             "minItems": 1 if refs else 0, "maxItems": len(refs) if refs else 0},
            "bottleneck_metric": ({"type": "string", "enum": [*bottleneck_metrics, "none"],
                                   "description": "Exact observed measurement key; 'none' only when blocked."}
                                  if bottleneck_metrics else string),
        }, "required": ["measured_bottleneck", "cause", "change", "expected_change_pct",
                         "validation", "history_refs", "bottleneck_metric"],
            "additionalProperties": False}
        required.append("experiment_plan")
    schema = {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
    if not analysis_only:
        schema["allOf"] = [{"if": {"properties": {"blocked": {"const": False}}},
                            "then": {"properties": {"experiment_plan": {"properties": {
                                "expected_change_pct": {"exclusiveMinimum": 0}}}}}}]
    return schema


def _allowed_history_refs(context):
    return sorted(set((context.get("candidate_contract") or {}).get("history_refs") or []))


def _read_json_message(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text)


def _normalize_history_refs(result, context, analysis_only=False):
    """Map evidence paths to canonical history IDs when that mapping is unique."""
    if analysis_only or not isinstance(result, dict):
        return result
    plan = result.get("experiment_plan")
    if not isinstance(plan, dict):
        return result
    allowed = _allowed_history_refs(context)
    allowed_set = set(allowed)
    by_iteration = {}
    for reference in allowed:
        try:
            iteration = int(reference.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            continue
        by_iteration.setdefault(iteration, []).append(reference)
    normalized = []
    for raw in plan.get("history_refs") or []:
        if raw in allowed_set:
            normalized.append(raw)
            continue
        text = str(raw)
        canonical = re.fullmatch(r"(.+):(\d+)", text)
        if canonical:
            candidate = f"{canonical.group(1)}:{int(canonical.group(2))}"
            if candidate in allowed_set:
                normalized.append(candidate)
                continue
        match = re.search(r"(?:^|/)iterations/(\d+)(?:/|$)", text)
        if match:
            matches = by_iteration.get(int(match.group(1)), [])
            if len(matches) == 1:
                normalized.append(matches[0])
    plan["history_refs"] = list(dict.fromkeys(normalized))
    return result


def _prompt(prompt, analysis_only=False, smoke=False, max_turns=20):
    if smoke:
        return ("Return the required JSON object with ready=true. Do not inspect or modify files. "
                "This verifies that Codex is using the local VibeForge Responses endpoint.")
    task = prompt
    if analysis_only:
        return ("MODE: analysis-only. Review only the assigned previous experiment and do not modify candidate "
                "files. The supplied brief's analysis_target contains the recorded decision and failure summary. "
                "When these identify the rejection, summarize them and submit; do not reread full profiler or "
                "model-loading logs. Keep the analysis within 200 words and distinguish facts from possible causes. "
                "Submit with the VibeForge MCP tool record_result exactly once, then stop. "
                "Use the inline brief when provided; otherwise read .vibeforge/brief.json. "
                "Read campaign evidence only for a specific missing fact, preferably with read_evidence pointer. "
                "This session does not select a new optimization, implement code or validate the current stable "
                "source as if it were the rejected candidate. No edit is required for successful analysis. "
                f"Use at most {max_turns} tool calls including submission.")
    else:
        task += (
            "\n\nMODE: optimization. Work only in the current candidate directory. Inspect and edit the candidate "
            "with Codex tools, then submit with the VibeForge MCP tool `record_result`. Do not claim that the candidate was "
            "accepted; the host validates the diff, plan, correctness, and performance."
        )
    # A bounded brief is the normal entry point. The complete host context is
    # available only for targeted queries; dumping it into the tool history can
    # consume most of a local model's context in a single turn.
    return task + (
        "\nUse the VibeForge MCP read_candidate tool for .vibeforge/brief.json and candidate source. "
        "The brief's candidate_source_index gives actual Python definition offsets; search_candidate finds "
        "literal source text. Read the relevant span instead of paging through entire files. "
        "Use replace_candidate for exact source edits, write_candidate for helper files, and check_candidate for Python or JSON syntax. "
        "These tools are confined to this candidate workspace. Do not keep retrying a broken shell. "
        "\n\nUse the inline bounded brief when supplied; otherwise read `.vibeforge/brief.json` once. The complete, unabridged VibeForge context "
        "is available at `.vibeforge/context.json` only for a targeted query of a specific missing field; "
        "never cat or print that entire file. Use the VibeForge MCP `search_history`, `list_evidence`, "
        "`read_evidence`, `read_telemetry`, `read_implementation`, and `profile_inference` tools when available; their budgets and evidence "
        "scope match the Qwen-Agent path. The submission schema is `.vibeforge/submission-schema.json`. "
        "The only authoritative result is a JSON "
        "object accepted by the MCP tool `record_result`; a final chat message does not submit a candidate. "
        f"Use no more than {max_turns} tool executions, including the submission command."
    )


def _probe_feedback_summary(context):
    """Keep the revision decision visible without copying profiler dumps."""
    feedback = context.get("candidate_feedback")
    if not feedback:
        return None
    result = {key: feedback[key] for key in (
        "kind", "candidate", "admission", "profile_id", "prompt_index", "mechanism",
        "feedback_state", "recommended_action", "needs_revision", "limitations", "error",
    ) if key in feedback}
    precheck = feedback.get("correctness_precheck") or {}
    result["correctness_precheck"] = {
        "passed": precheck.get("passed"), "scope": precheck.get("scope"),
        "checks": [{key: check[key] for key in (
            "case", "passed", "error", "input_lengths", "observed_batch_sizes", "batching_modes", "output_differences",
        ) if key in check} for check in precheck.get("checks", [])[:12]],
    }
    result["evidence_path"] = context.get("candidate_feedback_path")
    result["detail_policy"] = "Full probe and runtime profiles remain at evidence_path; read only missing details."
    result["measurement_scope"] = (
        "These measurements belong to the candidate ID above BEFORE this session's revision. "
        "They do not validate the edited successor; the host must test it after submission.")
    return result


def _brief(context, analysis_only=False, smoke=False):
    if smoke:
        return {"mode": "smoke", "iteration": context.get("iteration")}
    if analysis_only:
        previous = context.get("previous") or {}
        return {
            "schema_version": 2,
            "mode": "analysis_only",
            "campaign_id": context.get("campaign_id"),
            "iteration": context.get("iteration"),
            "analysis_target": {key: previous.get(key) for key in (
                "iteration", "parent", "candidate", "hypothesis", "decision", "error",
                "candidate_gate", "experiment_plan", "experiment_verification",
                "probe_feedback", "effect_scope", "complete", "validation_summary",
            ) if key in previous},
            "previous_analysis": (context.get("previous_analysis") or "")[:1800],
            "evidence_directory": (
                f"iterations/{previous['iteration']:04d}" if isinstance(previous.get("iteration"), int) else None),
            "instructions": context.get("instructions"),
            "agent_interface": agent_capabilities(context, analysis_only=True),
        }

    optimization = context.get("optimization_brief") or {}
    contract = context.get("candidate_contract") or {}
    cache_values = sorted(TRANSFORMERS_CACHE_IMPLEMENTATIONS)
    model_type = ((optimization.get("model_summary") or {}).get("model_type") or "")
    attention_values = (QWEN35_ATTENTION_BACKENDS if model_type.startswith("qwen3_5")
                        else TRANSFORMERS_ATTENTION_BACKENDS)
    source_index = candidate_index(context["workspace"]) if context.get("workspace") else {}
    return {
        "schema_version": 2,
        "mode": "refinement" if context.get("refinement_target") else "optimization",
        "campaign_id": context.get("campaign_id"),
        "iteration": context.get("iteration"),
        "knowledge": context.get("knowledge", []),
        "knowledge_scope": context.get("knowledge_scope", "project"),
        "transfer_hints": context.get("transfer_hints", []),
        "decision_packet": optimization,
        "requirements": context.get("requirements"),
        "candidate_contract": {
            "observation_paths": contract.get("observation_paths", []),
            "observed_bottleneck_metrics": sorted((context.get("performance_observation") or {}).get("measurements", {})),
            "allowed_history_refs": contract.get("history_refs", [])[-50:],
            "history_policy": (
                "Use only exact values from allowed_history_refs. The host rejects a proposal that omits "
                "history comparison when prior attempts exist."
            ),
        },
        "candidate_constraints": {
            "transformers_cache_implementations": cache_values,
            "cache_policy": "eligible_if_supported_by_mechanism_evidence",
            "attention_backends": sorted(attention_values),
            "rule": (
                "Values are syntax candidates, not performance evidence. Exact completed implementations remain "
                "deduplicated; a prior failure does not exclude every parameter or implementation in its family."
            ),
        },
        "candidate_feedback": _probe_feedback_summary(context),
        "refinement_target": context.get("refinement_target"),
        "candidate_engine_config": context.get("candidate_engine_config"),
        "candidate_files": source_index.get("files") or ["engine.py", "engine.json"],
        "candidate_source_index": source_index,
        "path_policy": "Candidate paths are relative to the workspace root, e.g. engine.json, not configs/engine.json.",
        "instructions": context.get("instructions"),
        "evidence_retrieval": (
            "Use facts already present in this brief. For a missing JSON field, use read_evidence with an exact "
            "path and pointer (for example /mechanism); read_candidate also supports pointer for host context. "
            "Use search_history only for a specific missing comparison. "
            "Do not print the full context.json."
        ),
        "agent_interface": agent_capabilities(context, analysis_only=False),
    }


def _source_inventory(root):
    return {path: checksum for path, checksum in inventory(root).items()
            if not path.startswith(".vibeforge/")}


def _changed_files(staging, workspace):
    before, after = _source_inventory(workspace), _source_inventory(staging)
    return sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))


def _candidate_diff(staging, workspace, changed):
    chunks = []
    for relative in changed:
        old_path, new_path = Path(workspace) / relative, Path(staging) / relative
        try:
            old = old_path.read_text(encoding="utf-8").splitlines(keepends=True) if old_path.is_file() else []
            new = new_path.read_text(encoding="utf-8").splitlines(keepends=True) if new_path.is_file() else []
        except UnicodeDecodeError:
            continue
        chunks.extend(difflib.unified_diff(old, new, fromfile=f"a/{relative}", tofile=f"b/{relative}"))
        if sum(map(len, chunks)) > 24000:
            chunks.append("\n[diff truncated]\n")
            break
    return "".join(chunks)


def _forced_tool_schema(analysis_only, bottleneck_metrics=None):
    string = {"type": "string"}
    properties = {
        "hypothesis": string, "previous_analysis": string, "summary": string,
        "blocked": {"type": "boolean"}, "blocked_reason": string,
        "reflection_iteration": {"type": "integer"},
        "reflection_hypothesis_status": {"type": "string", "enum": ["supported", "refuted", "inconclusive"]}, "reflection_explanation": string,
        "reflection_next_action": string,
        "reflection_evidence_paths_json": string,
    }
    if not analysis_only:
        properties.update({
            "plan_measured_bottleneck": string, "plan_cause": string, "plan_change": string,
            "plan_expected_change_pct": {"type": "number", "minimum": 0,
                                         "description": "Positive expected improvement magnitude in percent; 0 only if blocked."}, "plan_validation": string,
            "plan_history_refs_json": string,
            "plan_bottleneck_metric": ({"type": "string", "enum": [*bottleneck_metrics, "none"]}
                                       if bottleneck_metrics else string),
        })
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _expand_forced_submission(arguments, analysis_only):
    def array(name):
        value = arguments[name]
        parsed = json.loads(value) if isinstance(value, str) else value
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError(f"{name} must contain a JSON array of strings")
        return parsed

    result = {
        "hypothesis": arguments["hypothesis"],
        "previous_analysis": arguments["previous_analysis"],
        "summary": arguments["summary"],
        "blocked": arguments["blocked"],
        "blocked_reason": arguments["blocked_reason"],
        "reflection": {
            "iteration": arguments["reflection_iteration"],
            "hypothesis_status": arguments["reflection_hypothesis_status"],
            "explanation": arguments["reflection_explanation"],
            "next_action": arguments["reflection_next_action"],
            "evidence_paths": array("reflection_evidence_paths_json"),
        },
    }
    if not analysis_only:
        result["experiment_plan"] = {
            "measured_bottleneck": arguments["plan_measured_bottleneck"],
            "cause": arguments["plan_cause"],
            "change": arguments["plan_change"],
            "expected_change_pct": arguments["plan_expected_change_pct"],
            "validation": arguments["plan_validation"],
            "history_refs": array("plan_history_refs_json"),
            "bottleneck_metric": arguments["plan_bottleneck_metric"],
        }
    return result


def _bind_forced_candidate_status(result, changed, analysis_only):
    """Derive workflow blockage from whether Codex produced a candidate diff."""
    if analysis_only or not isinstance(result, dict):
        return result
    if changed:
        # Post-change evidence cannot exist until the host accepts the proposal
        # and runs its frozen gates. A local model must not use that expected
        # absence to suppress an otherwise testable candidate.
        result["blocked"] = False
        result["blocked_reason"] = ""
    else:
        result["blocked"] = True
        reason = ("Codex produced no candidate file changes. This establishes incomplete implementation, "
                  "not an environment failure or a measured performance result.")
        result["blocked_reason"] = reason
        result["summary"] = reason
        result["hypothesis"] = "No optimization candidate was implemented."
        if isinstance(result.get("reflection"), dict):
            result["reflection"].update(hypothesis_status="inconclusive", explanation=reason,
                                        next_action="Complete one candidate edit and submit it for host validation.")
    return result


def _forced_submission(url, context, staging, workspace, analysis_only, final_message, timeout):
    """Use a named function call to close a Codex pass that omitted its MCP call."""
    history_refs = _allowed_history_refs(context)
    metrics = sorted((context.get("performance_observation") or {}).get("measurements", {}))
    schema = _result_schema(analysis_only, False, history_refs, metrics)
    tool_schema = _forced_tool_schema(analysis_only, metrics)
    changed = [] if analysis_only else _changed_files(staging, workspace)
    diff = "" if analysis_only else _candidate_diff(staging, workspace, changed)
    mode = "analysis-only" if analysis_only else "optimization"
    evidence = {
        "mode": mode,
        "iteration": context.get("iteration"),
        "requirements": context.get("requirements"),
        "changed_files": changed,
        "candidate_diff": diff,
        "codex_final_message": final_message[:6000],
        "previous": context.get("previous"),
        "current_evidence": _brief(context, analysis_only=analysis_only),
        "allowed_history_refs": history_refs,
    }
    if not analysis_only and not changed:
        evidence["instruction"] = (
            "No candidate files changed. Submit blocked=true and explain that Codex produced no concrete edit. "
            "Previous failures are historical; do not attribute this session to an old environment error.")
    else:
        evidence["instruction"] = (
            "Submit an evidence-grounded result for the recorded analysis or candidate diff. Do not invent edits or "
            "measurements. When changed_files is non-empty, submit blocked=false: post-change correctness and "
            "performance measurements are intentionally produced by the host only after this submission. "
            "In optimization mode, hypothesis and experiment_plan must describe the CURRENT diff and current "
            "measurements. The previous experiment is historical and belongs only in previous_analysis/reflection. "
            "Do not reuse its hypothesis for an unrelated edit. During refinement, candidate_diff is the new "
            "revision delta, while refinement_target.source_patch is the existing candidate. Describe their "
            "combined change. candidate_feedback measured the pre-revision candidate, not the new edit.")
    payload = {
        "model": (context.get("agent") or {}).get("model") or (context.get("model") or {}).get("name"),
        "messages": [
            {"role": "system", "content": (
                "You are the final VibeForge submission stage. You must call record_result exactly once. "
                "Use only the supplied evidence and keep unmeasured outcomes inconclusive.")},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
        ],
        "tools": [{"type": "function", "function": {
            "name": "record_result",
            "description": ("Submit the authoritative VibeForge result. Fields ending in _json must be valid "
                            "JSON arrays of strings, such as []."),
            "parameters": tool_schema,
        }}],
        "tool_choice": {"type": "function", "function": {"name": "record_result"}},
        "parallel_tool_calls": False,
        "max_tokens": int((context.get("agent") or {}).get("max_new_tokens", 512)),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = EngineClient(url, timeout=timeout).request("v1/chat/completions", payload)
    calls = ((response.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "record_result":
        raise ValueError("Structured submission stage did not call record_result exactly once")
    arguments = calls[0]["function"].get("arguments", "{}")
    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    validate(arguments, tool_schema)
    result = _expand_forced_submission(arguments, analysis_only)
    result = _normalize_history_refs(result, context, analysis_only)
    result = _bind_forced_candidate_status(result, changed, analysis_only)
    validate(result, schema)
    return result


def _run_codex_process(command, request, *, cwd, environment, timeout):
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=cwd, env=environment, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(request, timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr), False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        stderr += f"\nCodex work pass exceeded {timeout} seconds."
        return subprocess.CompletedProcess(command, 124, stdout, stderr), True


def _event_kind(event):
    """Return the most specific public Codex JSONL event/item kind."""
    candidates = [event.get("type")]
    for key in ("item", "event", "msg"):
        nested = event.get(key)
        if isinstance(nested, dict):
            candidates.append(nested.get("type"))
    return next((value for value in reversed(candidates) if isinstance(value, str)), "unknown")


def _usage_snapshots(value, path=""):
    """Find token counters without depending on one Codex CLI event version."""
    snapshots = []
    if isinstance(value, dict):
        keys = set(value)
        token_keys = keys & {
            "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
            "reasoning_output_tokens", "reasoning_tokens",
        }
        if token_keys:
            snapshots.append({"path": path or "$", **{
                key: value[key] for key in sorted(token_keys)
                if isinstance(value[key], (int, float)) and not isinstance(value[key], bool)
            }})
        for key, child in value.items():
            snapshots.extend(_usage_snapshots(child, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            snapshots.extend(_usage_snapshots(child, f"{path}[{index}]"))
    return [snapshot for snapshot in snapshots if len(snapshot) > 1]


def summarize_codex_events(stdout):
    """Summarize Codex-owned context management while retaining raw JSONL."""
    events, malformed = [], 0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(value, dict):
            events.append(value)
    kinds = [_event_kind(event) for event in events]
    compactions = []
    for index, (event, kind) in enumerate(zip(events, kinds)):
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        notice = item.get("message", "") if item.get("type") == "error" else ""
        if ("compact" in kind.casefold()
                or any("compact" in str(event.get(key, "")).casefold()
                       for key in ("event_type", "method", "name"))
                or "compaction" in notice.casefold()):
            compactions.append({"event_index": index, "kind": kind,
                                "source": "codex_notice" if notice else "event_type"})
    usage = []
    for index, event in enumerate(events):
        for snapshot in _usage_snapshots(event):
            usage.append({"event_index": index, "kind": kinds[index], **snapshot})
    return {
        "schema_version": 1,
        "context_owner": "codex",
        "vibeforge_prompt_compaction": False,
        "auto_compaction_policy": "Codex automatic compaction with host capacity reserves",
        "event_count": len(events),
        "malformed_line_count": malformed,
        "event_types": dict(sorted(Counter(kinds).items())),
        "compaction_count": len(compactions),
        "compaction_events": compactions,
        "usage_snapshots": usage,
    }


def _structured_recovery(context, analysis_only, error):
    """Return an explicit host recovery record when a local model ignores the schema."""
    previous = context.get("previous") or {}
    target = previous.get("iteration") if analysis_only else context.get("iteration", 0)
    target = target if isinstance(target, int) else 0
    evidence = f"iterations/{target:04d}/decision.json" if analysis_only and target else ""
    summary = "Codex completed, but the local model did not return the required structured JSON."
    result = {
        "hypothesis": f"Review iteration {target} execution failure." if analysis_only
                      else "No structured optimization proposal was produced.",
        "previous_analysis": (
            f"Iteration {target}: the recorded attempt ended without a valid structured Agent result. "
            "No candidate performance conclusion can be drawn; the outcome is inconclusive."
            if analysis_only else (context.get("previous_analysis") or "")[:1800]
        ),
        "summary": summary,
        "blocked": True,
        "blocked_reason": error,
        "reflection": {
            "iteration": target,
            "hypothesis_status": "inconclusive",
            "explanation": ("The model output could not be validated against the frozen result schema; "
                            "no causal or performance claim is accepted."),
            "next_action": "Start a fresh bounded Codex session and require a structured final response.",
            "evidence_paths": [evidence] if evidence else [],
        },
    }
    if not analysis_only:
        history_refs = _allowed_history_refs(context)
        result["experiment_plan"] = {
            "measured_bottleneck": "No validated proposal",
            "cause": "Local model omitted the required structured final response",
            "change": "No candidate is submitted",
            "expected_change_pct": 0,
            "validation": "Skip candidate evaluation and start a fresh Agent session",
            "history_refs": history_refs[-1:] if history_refs else [],
            "bottleneck_metric": "none",
        }
    return result


def _bind_analysis_target(result, context, analysis_only):
    """Bind reflection identity to the host-selected pending experiment."""
    if not analysis_only or not isinstance(result, dict):
        return result
    target = (context.get("previous") or {}).get("iteration")
    reflection = result.get("reflection")
    if isinstance(target, int) and isinstance(reflection, dict):
        reflection["iteration"] = target
        if reflection.get("hypothesis_status") not in {"supported", "refuted", "inconclusive"}:
            reflection["hypothesis_status"] = "inconclusive"
        reason = ((context.get("previous") or {}).get("decision") or {}).get("reason")
        suffix = "correctness/report.json" if reason == "correctness_failed" else "decision.json"
        reflection["evidence_paths"] = [f"iterations/{target:04d}/{suffix}"]
    return result


def _sync_candidate(staging, workspace, trace):
    before = inventory(workspace)
    after = inventory(staging)
    removed = sorted(set(before) - set(after))
    if removed:
        raise RuntimeError("Codex cannot delete candidate files: " + ", ".join(removed[:5]))
    reserved_roots = {
        ".vibeforge", "campaigns", "final-analysis", "iterations",
        "knowledge-drafts", "logs", "reports", "var",
    }
    reserved = sorted(name for name in after if Path(name).parts[0] in reserved_roots
                      and before.get(name) != after[name])
    if reserved:
        raise RuntimeError("Codex cannot add or modify VibeForge artifacts in the candidate: "
                           + ", ".join(reserved[:5]))
    updated = []
    for relative, checksum in after.items():
        if before.get(relative) == checksum:
            continue
        source = Path(staging) / relative
        if source.stat().st_size > 512 * 1024:
            raise RuntimeError(f"Codex candidate file exceeds 512 KiB: {relative}")
        try:
            content = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"Codex candidate files must be UTF-8 text: {relative}") from error
        atomic_write(Path(workspace) / relative, content)
        updated.append(relative)
    append_record(trace, {"event": "codex_candidate_synced", "updated": updated})


def run_codex_agent(url, context, workspace, trace, prompt, analysis_only=False, smoke=False):
    settings = context.get("agent", {})
    executable = settings.get("executable", "codex")
    resolved = shutil.which(executable)
    if not resolved:
        raise RuntimeError(f"Codex executable not found: {executable}")
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    artifact_root = Path(trace).resolve().parent
    artifact_root.mkdir(parents=True, exist_ok=True)
    suffix = "" if Path(trace).name == "commands.jsonl" else "-" + Path(trace).stem
    schema_path = artifact_root / f"codex{suffix}-output-schema.json"
    output_path = artifact_root / f"codex{suffix}-last-message.json"
    submission_path = artifact_root / f"codex{suffix}-submission.json"
    output_path.unlink(missing_ok=True)
    submission_path.unlink(missing_ok=True)
    history_refs = _allowed_history_refs(context)
    metrics = sorted((context.get("performance_observation") or {}).get("measurements", {}))
    result_schema = _result_schema(analysis_only, smoke, history_refs, metrics)
    write_json(schema_path, result_schema)
    model = settings.get("model") or context.get("model", {}).get("name") or "vibeforge-local"
    context_window = int(context.get("model", {}).get("max_context", 4096))
    output_reserve = int(settings.get("max_new_tokens", 512))
    tool_output_limit = min(4096, max(512, output_reserve))
    safety_reserve = max(256, min(1024, context_window // 20))
    compact_limit = context_window - output_reserve - tool_output_limit - safety_reserve
    if compact_limit <= 1024:
        raise ValueError(
            "Codex context window is too small for output, tool-result and compaction reserves")
    provider = "vibeforge_local"
    session_id = uuid.uuid4().hex
    request = _prompt(prompt, analysis_only, smoke, settings.get("max_turns", 20))
    brief_text = json.dumps(_brief(context, analysis_only, smoke), ensure_ascii=False, separators=(",", ":"))
    inline_brief = ""
    if (not smoke and context_window >= 16384
            and len(brief_text.encode("utf-8")) <= min(32000, context_window)):
        inline_brief = (
            "\n\nThe complete bounded brief is supplied below. Do not reread brief.json. "
            + ("Summarize the assigned experiment's recorded outcome and call record_result; read another "
               "evidence field only if the decision or failure facts are missing.\n" if analysis_only else
               "Inspect the relevant implementation, edit, check, and submit. The full context remains "
               "available for targeted missing facts.\n")
            + brief_text)
        request += inline_brief
    append_record(trace, {"event": "codex_started", "session_id": session_id, "model": model,
                          "context_window": context_window, "context_owner": "codex",
                          "vibeforge_prompt_compaction": False,
                          "codex_auto_compact_token_limit": compact_limit,
                          "codex_tool_output_token_limit": tool_output_limit,
                          "workspace": str(workspace), "analysis_only": analysis_only, "smoke": smoke})
    environment = dict(os.environ)
    environment["NO_COLOR"] = "1"
    result = None
    recovery_error = None
    submitted_via_tool = False
    submission_origin = None
    with tempfile.TemporaryDirectory(prefix="vibeforge-codex-candidate-") as temporary:
        staging = Path(temporary) / "workspace"
        if any(workspace.iterdir()):
            create_workspace(workspace, staging)
        else:
            staging.mkdir()
        context_mount = staging / ".vibeforge"
        context_mount.mkdir()
        write_json(context_mount / "context.json", {**context, "analysis_only": analysis_only})
        write_json(context_mount / "brief.json", _brief(context, analysis_only, smoke))
        write_json(context_mount / "submission-schema.json", result_schema)
        output_arguments = ["--json", "--output-last-message", str(output_path)]
        if smoke:
            output_arguments[1:1] = ["--output-schema", str(schema_path)]
        mcp_arguments = []
        if not smoke:
            server_arguments = [
                # Codex filters inherited environment variables for MCP. Bind
                # the bridge to this controller's import root explicitly, so
                # another editable installation cannot supply stale tools.
                "-c", ("import sys; sys.path.insert(0, "
                       + repr(str(Path(__file__).resolve().parents[2]))
                       + "); from vibeforge.agent.codex_submit import main; main()"),
                "--schema", str(schema_path), "--output", str(submission_path),
                "--context", str(context_mount / "context.json"),
                "--workspace", str(staging), "--trace", str(trace),
            ]
            enabled_tools = ["record_result", "read_candidate", "search_candidate"]
            if not analysis_only:
                enabled_tools.extend(["replace_candidate", "write_candidate", "check_candidate"])
            if context.get("evidence_root"):
                enabled_tools.extend(["list_evidence", "read_evidence", "search_history"])
            if context.get("telemetry_session") and context.get("engine_url"):
                enabled_tools.append("read_telemetry")
            profiling = (context.get("loop_protocol", {}).get("profiling") or {})
            if (context.get("candidate_contract") and context.get("engine_url") and not analysis_only
                    and profiling.get("enabled", True)):
                enabled_tools.extend(["profile_inference", "read_implementation"])
            mcp_arguments = [
                "-c", f"mcp_servers.vibeforge_submit.command={json.dumps(sys.executable)}",
                "-c", f"mcp_servers.vibeforge_submit.args={json.dumps(server_arguments)}",
                "-c", "mcp_servers.vibeforge_submit.required=true",
                "-c", ("mcp_servers.vibeforge_submit.enabled_tools="
                       f"{json.dumps(enabled_tools, separators=(',', ':'))}"),
                "-c", "mcp_servers.vibeforge_submit.default_tools_approval_mode=\"approve\"",
            ]
        command = [
            resolved, "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
            "--skip-git-repo-check", "--sandbox", "read-only" if smoke else "workspace-write",
            *output_arguments,
            "--cd", str(staging), "--model", model,
            "-c", f"model_provider={json.dumps(provider)}",
            "-c", f"model_context_window={context_window}",
            "-c", f"model_auto_compact_token_limit={compact_limit}",
            "-c", f"tool_output_token_limit={tool_output_limit}",
            "-c", "approval_policy=\"never\"",
            "-c", "features.multi_agent=false",
            *(["-c", "features.shell_tool=false"]
              if context.get("knowledge_scope") == "current_campaign" else []),
            "-c", "web_search=\"disabled\"",
            "-c", f"model_providers.{provider}.name=\"VibeForge local stable engine\"",
            "-c", f"model_providers.{provider}.base_url={json.dumps(url.rstrip('/') + '/v1')}",
            "-c", f"model_providers.{provider}.wire_api=\"responses\"",
            "-c", f"model_providers.{provider}.requires_openai_auth=false",
            "-c", f"model_providers.{provider}.supports_websockets=false",
            *mcp_arguments,
            "-",
        ]
        # Analysis is read-only and a completed pass already contains all of
        # the evidence needed by the strict submission fallback. Re-running
        # optimization work prompts cannot improve it and only delays cleanup.
        max_passes = 1 if smoke or analysis_only else int(settings.get("max_passes", 3))
        total_timeout = float(settings.get("timeout_seconds", 180))
        # Leave time for schema finalization and file synchronization before
        # the supervisor's outer agent deadline terminates this worker.
        work_budget = max(1.0, total_timeout - min(120.0, total_timeout * 0.2))
        pass_timeout = (min(120.0, total_timeout) if smoke else
                        min(300.0, work_budget) if analysis_only else work_budget / max_passes)
        stdout_parts, stderr_parts = [], []
        completed = None
        final_message = ""
        for pass_index in range(max_passes):
            pass_request = request
            pass_command = list(command)
            if pass_index:
                pending_changes = _changed_files(staging, workspace)
                source = {}
                for relative in ("engine.py", "engine.json"):
                    path = staging / relative
                    if path.is_file():
                        source[relative] = path.read_text(encoding="utf-8")[:6000]
                pass_request = (
                    f"IMPLEMENTATION STAGE {pass_index + 1}/{max_passes}. Exploration is complete. "
                    "The previous pass ended without an edit. Use the supplied source and evidence now. "
                    "Implement one evidence-grounded hypothesis, then check_candidate and record_result. "
                    "The inline source is only a prefix, not the permitted search space: use the source index "
                    "and search_candidate/read_candidate to inspect a missing implementation span. "
                    "Do not select a configuration edit merely because it appears in the prefix. "
                    "Do not narrate future work or read entire files. If no justified edit is possible, call "
                    "record_result with blocked=true now.\nPrevious final message: " + final_message[:1500]
                    + "\nCurrent candidate source (first 6000 characters per file): "
                    + json.dumps(source, ensure_ascii=False)
                ) + (inline_brief or "\nRead .vibeforge/brief.json only for the missing decision evidence.")
                if pending_changes:
                    pass_request = (
                        "SUBMISSION STAGE. The previous pass already edited these candidate files: "
                        + json.dumps(pending_changes) + ". Preserve the existing hypothesis and edit. "
                        "Check the changed files and call record_result with the actual change now. "
                        "Do not add another optimization or claim missing edits.\nExisting patch:\n"
                        + _candidate_diff(staging, workspace, pending_changes)
                        + inline_brief)
                for index, value in enumerate(pass_command):
                    if value.startswith("mcp_servers.vibeforge_submit.enabled_tools="):
                        stage_tools = ["read_candidate", "search_candidate", "check_candidate", "record_result"]
                        if not pending_changes:
                            stage_tools.insert(1, "replace_candidate")
                            stage_tools.insert(2, "write_candidate")
                        pass_command[index] = ('mcp_servers.vibeforge_submit.enabled_tools='
                                               + json.dumps(stage_tools, separators=(',', ':')))
            completed, timed_out = _run_codex_process(
                pass_command, pass_request, cwd=staging, environment=environment, timeout=pass_timeout)
            stdout_parts.append(completed.stdout)
            stderr_parts.append(completed.stderr)
            atomic_write(artifact_root / f"codex{suffix}-pass-{pass_index + 1}.jsonl", completed.stdout)
            atomic_write(artifact_root / f"codex{suffix}-pass-{pass_index + 1}.stderr", completed.stderr)
            append_record(trace, {"event": "codex_pass_completed", "session_id": session_id,
                                  "pass": pass_index + 1, "exit_code": completed.returncode,
                                  "timed_out": timed_out,
                                  "changed_files": _changed_files(staging, workspace)})
            if output_path.is_file():
                final_message = output_path.read_text(encoding="utf-8", errors="replace")
            changed = _changed_files(staging, workspace)
            if timed_out and submission_path.is_file():
                # MCP submission is authoritative. A model that continues
                # talking afterward must not erase a valid committed result.
                try:
                    validate(read_json(submission_path), result_schema)
                except (OSError, ValueError, ValidationError):
                    pass
                else:
                    completed.returncode = 0
                    append_record(trace, {"event": "codex_timeout_after_submission",
                                          "session_id": session_id, "pass": pass_index + 1})
            if submission_path.is_file() or (changed and not timed_out):
                break
            if timed_out and analysis_only:
                # Analysis has no candidate state to preserve. The host-owned
                # forced tool stage can close the reflection from the frozen
                # previous-experiment evidence after Codex fails to exit.
                completed.returncode = 0
                append_record(trace, {
                    "event": "codex_analysis_timeout_fallback",
                    "session_id": session_id,
                    "pass": pass_index + 1,
                })
                break
            if timed_out and changed and pass_index == max_passes - 1:
                completed.returncode = 0
                break
        completed.stdout = "\n".join(stdout_parts)
        completed.stderr = "\n".join(stderr_parts)
        if not completed.returncode and not smoke and submission_path.is_file():
            try:
                result = read_json(submission_path)
                result = _normalize_history_refs(result, context, analysis_only)
                validate(result, result_schema)
                submitted_via_tool = True
                submission_origin = "mcp_submit"
            except (OSError, ValueError, ValidationError) as error:
                recovery_error = f"{type(error).__name__}: {error}"
                result = _structured_recovery(context, analysis_only, recovery_error)
                validate(result, result_schema)
        if not completed.returncode and not smoke and result is None:
            try:
                result = _forced_submission(
                    url, context, staging, workspace, analysis_only, final_message,
                    settings.get("request_timeout", settings.get("timeout_seconds", 180)),
                )
                submission_origin = "forced_tool_submit"
                write_json(submission_path, result)
            except (OSError, RuntimeError, ValueError, ValidationError) as error:
                recovery_error = f"{type(error).__name__}: {error}"
                result = _structured_recovery(context, analysis_only, recovery_error)
                validate(result, result_schema)
        result = _bind_analysis_target(result, context, analysis_only)
        # Host context is an ephemeral read-only input, never candidate source.
        shutil.rmtree(context_mount)
        if not completed.returncode and result is None and smoke:
            try:
                if not output_path.is_file():
                    raise ValueError("Codex completed without a structured last message")
                result = _read_json_message(output_path)
                validate(result, result_schema)
            except (OSError, ValueError, ValidationError) as error:
                recovery_error = f"{type(error).__name__}: {error}"
        if (not completed.returncode and result is not None and not analysis_only and not smoke
                and not result.get("blocked", False)):
            _sync_candidate(staging, workspace, trace)
    event_path = artifact_root / f"codex{suffix}-events.jsonl"
    atomic_write(event_path, completed.stdout)
    context_summary = summarize_codex_events(completed.stdout)
    context_summary.update(session_id=session_id, model=model, context_window=context_window,
                           prompt_characters=len(request),
                           codex_auto_compact_token_limit=compact_limit,
                           codex_tool_output_token_limit=tool_output_limit,
                           output_token_reserve=output_reserve,
                           safety_token_reserve=safety_reserve)
    write_json(artifact_root / f"codex{suffix}-context.json", context_summary)
    if completed.returncode:
        append_record(trace, {"event": "codex_failed", "session_id": session_id,
                              "exit_code": completed.returncode, "stderr": completed.stderr[-4000:]})
        raise RuntimeError(f"Codex exited with code {completed.returncode}: {completed.stderr[-1000:]}")
    if result is None:
        raise RuntimeError(recovery_error or "Codex completed without a structured last message")
    if smoke:
        if not result.get("ready"):
            raise RuntimeError("Codex readiness response was not successful")
        return {"session_id": session_id, "mode": "codex", "complete": True, **result}
    result.update(session_id=session_id, mode="codex", complete=True)
    result.setdefault("submission_origin", "host_recovery" if recovery_error else (
        submission_origin or ("mcp_submit" if submitted_via_tool else "agent")))
    if recovery_error:
        append_record(trace, {"event": "codex_structured_output_recovered", "session_id": session_id,
                              "error": recovery_error})
    append_record(trace, {"event": "codex_completed", "session_id": session_id,
                          "blocked": result.get("blocked", False),
                          "compaction_count": context_summary["compaction_count"],
                          "usage_snapshots": context_summary["usage_snapshots"]})
    return result
