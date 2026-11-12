"""Shared bounded stable-engine profiling tool for Agent adapters."""

import time
from pathlib import Path

from vibeforge.contracts.workload import workload_profiles
from vibeforge.engine.client import EngineClient
from vibeforge.observability.logs import append_record
from vibeforge.storage.atomic import write_json


def read_implementation(context, trace, arguments):
    from vibeforge.storage.atomic import digest
    from vibeforge.versioning.artifacts import source_hash
    report = EngineClient(context["engine_url"], 30).implementation(**arguments)
    parent = context["candidate_contract"]["parent_source"]
    if report.get("source_hash") != source_hash(parent):
        raise ValueError("Implementation source does not belong to the active stable release")
    relative = f"observations/{context['stable_ref']}/implementation-{digest(arguments)[:16]}.json"
    write_json(Path(context["evidence_root"]) / relative, report)
    append_record(trace, {"tool": "read_implementation", "arguments": arguments, "result_path": relative})
    return {"evidence_path": relative, **report}


def profile_inference(context, trace, arguments, calls):
    """Collect one host-owned diagnostic profile under the frozen session budget."""
    index = arguments["prompt_index"]
    profiles = workload_profiles(context["workload"])
    default_profile = context["workload"].get("observation_profile", profiles[0]["id"])
    selected = next((profile for profile in profiles
                     if profile["id"] == arguments.get("profile_id", default_profile)), None)
    if selected is None:
        raise ValueError("Unknown frozen workload profile")
    prompts = selected["prompts"]
    settings = (context.get("loop_protocol", {}).get("profiling")
                or context.get("profiling") or context.get("observation", {}))
    if not settings.get("enabled", True):
        raise ValueError("Diagnostic profiling is disabled for this campaign")
    limit = settings.get("max_profiles_per_session", 2)
    if type(index) is not int or not 0 <= index < len(prompts) or len(calls) >= limit:
        raise ValueError("Invalid prompt index or diagnostic profile budget exhausted")
    calls.append({"profile_id": selected["id"], "prompt_index": index})
    report = EngineClient(
        context["engine_url"], min(
            context["agent"].get("request_timeout", 180),
            settings.get("timeout_seconds", 300),
        )
    ).profile(
        prompt=prompts[index],
        max_new_tokens=min(selected["max_new_tokens"], settings.get("max_new_tokens", 128), 128),
    )
    report.update(
        stable_ref=context["stable_ref"], profile_hash=context["profile_hash"],
        workload=context["workload"], prompt_index=index, profile_id=selected["id"],
        observed_at=time.time(),
    )
    relative = f"observations/{context['stable_ref']}/agent-{context['iteration']}-{len(calls)}.json"
    write_json(Path(context["evidence_root"]) / relative, report)
    contract = context.get("candidate_contract") or {}
    contract.setdefault("observation_paths", []).append(relative)
    append_record(trace, {"tool": "profile_inference", "arguments": arguments,
                          "result_path": relative})
    return {"evidence_path": relative, "profile": report}
