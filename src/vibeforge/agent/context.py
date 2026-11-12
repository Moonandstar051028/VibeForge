"""Rebuild restart context from committed state and evidence paths."""

import shutil

from vibeforge.storage.atomic import write_json, read_json
from vibeforge.storage.paths import confined
from vibeforge.agent.backend_policy import selected_attention_backends
from vibeforge.agent.brief import build_optimization_brief
from vibeforge.agent.candidate import (
    history_reference, observation_benefit_bounds, strategy_from_plan,
)
from vibeforge.agent.memory import (
    archived_candidates, experience_ledger, older_history_summary,
    refresh_memory, reusable_candidate_result,
)


def transfer_hints(database, paths, campaign_id, config, limit=6):
    """Carry relevant experiments across protocol changes as hints, never exclusions."""
    if config.get("knowledge", {}).get("scope") == "current_campaign":
        return []
    hints = []
    for campaign in reversed(database.list_campaigns()):
        other = campaign.get("campaign_id")
        manifest = paths.campaign(other) / "manifest.json" if other else None
        if not other or other == campaign_id or not manifest.is_file():
            continue
        previous_config = read_json(manifest)
        if (previous_config.get("model", {}).get("path") != config["model"].get("path")
                or previous_config.get("model", {}).get("name") != config["model"].get("name")
                or previous_config.get("precision") != config["precision"]):
            continue
        for row in reversed(database.iterations(other)):
            decision = row.get("decision") or {}
            if (not row.get("candidate") or not row.get("complete")
                    or not reusable_candidate_result(decision, row.get("experiment_verification"))):
                continue
            relative = f"memory/transfer/{other}/{row['iteration']:04d}"
            source = paths.campaign(other) / "iterations" / f"{row['iteration']:04d}"
            destination = paths.campaign(campaign_id) / relative
            for name in ("decision.json", "change.patch", "verification.json", "correctness/report.json"):
                if (source / name).is_file():
                    (destination / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / name, destination / name)
            hints.append({
                "campaign_id": other, "iteration": row["iteration"],
                "hypothesis": (row.get("hypothesis") or "")[:250],
                "decision": decision, "evidence_path": relative,
                "same_protocol": previous_config.get("config_hash") == config.get("config_hash"),
                "scope": "Same model path/name and precision. Hardware/workload/source may differ. "
                         "Use only as an investigation hint; fresh evidence and all gates remain mandatory. "
                         "This entry is not an allowed history_ref unless separately listed in the contract.",
            })
            if len(hints) >= limit:
                return hints
    return hints


def _database_candidates(database, paths, campaign_id, stable_ref, profile_hash, config_hash):
    """Index completed equivalent experiments from ordinary prior campaigns."""
    result = []
    for campaign in database.list_campaigns():
        other_id = campaign.get("campaign_id")
        if (not other_id or other_id == campaign_id or campaign.get("config_hash") != config_hash
                or campaign.get("profile_hash") != profile_hash):
            continue
        for row in database.iterations(other_id):
            gate = row.get("candidate_gate") or {}
            decision = row.get("decision") or {}
            verification = row.get("experiment_verification") or {}
            if (row.get("parent") != stable_ref or not row.get("candidate") or not row.get("complete")
                    or not gate.get("valid") or not gate.get("fingerprint")
                    or not reusable_candidate_result(decision, verification)):
                continue
            source = paths.campaign(other_id) / "iterations" / f"{row['iteration']:04d}"
            relative_evidence = f"memory/prior-campaigns/{other_id}/iterations/{row['iteration']:04d}"
            evidence = confined(paths.campaign(campaign_id), relative_evidence)
            for name in ("decision.json", "change.patch", "memory.json", "correctness/report.json",
                         "verification.json", "benchmark/report.json", "benchmarks/report.json"):
                source_file = source / name
                if source_file.is_file():
                    destination = evidence / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, destination)
            result.append({
                "campaign_id": other_id, "iteration": row["iteration"],
                "history_ref": history_reference(other_id, row["iteration"]),
                "fingerprint": gate["fingerprint"], "candidate": row["candidate"],
                "effect_fingerprint": row.get("effect_fingerprint"),
                "effect_scope": row.get("effect_scope"),
                "hypothesis": (row.get("hypothesis") or "")[:240], "decision": decision,
                "experiment_plan": row.get("experiment_plan"),
                "verification": verification,
                "evidence_path": relative_evidence,
            })
    return result


def create_context(database, paths, campaign_id, config, workspace, iteration, analysis_only=False):
    isolated = config.get("knowledge", {}).get("scope") == "current_campaign"
    state = database.get(campaign_id)
    history = [row for row in database.iterations(campaign_id) if row["iteration"] < iteration]
    previous = next((row for row in reversed(history) if row["iteration"] < iteration), None)
    pending = [row for row in history if row["iteration"] < iteration and row.get("analysis_pending")]
    if pending and analysis_only:
        previous = pending[0]
    previous_analysis = ""
    if previous:
        analysis_path = paths.campaign(campaign_id) / "iterations" / f"{previous['iteration']:04d}" / "analysis.md"
        if analysis_path.is_file():
            previous_analysis = analysis_path.read_text()[:6000]
        correctness_path = analysis_path.parent / "correctness" / "report.json"
        if correctness_path.is_file():
            report = read_json(correctness_path)
            failures = [check for check in report.get("checks", []) if not check.get("passed")]
            previous = {**previous, "validation_summary": {
                "evidence_path": str(correctness_path.relative_to(paths.campaign(campaign_id))),
                "passed": report.get("passed"), "failed_check_count": len(failures),
                "failed_checks": [{key: check[key] for key in (
                    "case", "passed", "generation_passed", "repeat_passed", "error", "numerical",
                    "continuation_numerical") if key in check} for check in failures[:12]],
                "interpretation": "Recorded check outcomes only; causal explanation remains unverified.",
            }}
    profile_hash = state.get("version_profiles", {}).get(state.get("stable_ref"), state["profile_hash"])
    profile = read_json(paths.data / "profiles" / f"{profile_hash}.json")
    directory = paths.campaign(campaign_id)
    memory_entries = refresh_memory(database, campaign_id, directory)
    recent_history = {row["iteration"] for row in history[-6:]}
    # Selecting older context by the previous hypothesis creates a feedback
    # loop: once one strategy is attempted, the next prompt is dominated by
    # semantically similar attempts. Recent records plus the complete compact
    # experience ledger provide history without anchoring retrieval this way.
    history_context = []
    for row in history:
        if row["iteration"] not in recent_history:
            continue
        candidate = row.get("candidate")
        release = paths.release(candidate) if candidate else None
        strategy = strategy_from_plan(row.get("experiment_plan"))
        history_context.append({
            "iteration": row["iteration"], "candidate": candidate,
            "history_ref": history_reference(campaign_id, row["iteration"]),
            "hypothesis": row.get("hypothesis"), "decision": row.get("decision"),
            "error": row.get("error"),
            "attention_backends": sorted(selected_attention_backends(release)) if release else [],
            "evidence_path": f"iterations/{row['iteration']:04d}",
            "hypothesis_status": (row.get("reflection") or {}).get("hypothesis_status"),
            "change": ((row.get("experiment_plan") or {}).get("change") or "")[:240],
            **strategy,
        })
    runtime = read_json(workspace / "engine.json")
    stable_config_path = paths.release(state["stable_ref"]) / "engine.json" if state.get("stable_ref") else None
    stable_runtime = read_json(stable_config_path) if stable_config_path and stable_config_path.is_file() else {}
    config_keys = (
        "backend", "model_loader", "device", "dtype", "use_cache",
        "cache_implementation", "attn_implementation", "request_scheduler",
        "microbatch_max_size", "microbatch_wait_ms", "num_threads",
    )
    value = {
        "context_schema_version": 4,
        "knowledge_scope": "current_campaign" if isolated else "project",
        "campaign_id": campaign_id, "iteration": iteration, "stable_ref": state.get("stable_ref"),
        "config_hash": config["config_hash"], "profile_hash": profile_hash,
        "precision": read_json(workspace / "engine.json").get("precision", config["precision"]),
        "stable_engine_config": {key: stable_runtime[key] for key in config_keys if key in stable_runtime},
        "candidate_engine_config": {key: runtime[key] for key in config_keys if key in runtime},
        "optimization": config["optimization"], "requirements": config["requirements"],
        "engine_adapter": config["engine"],
        "modification_surfaces": {
            "allowed": config["optimization"]["allowed_surfaces"],
            "runtime_config": ["use_cache", "cache_implementation", "attn_implementation", "num_threads"],
            "scheduler": {"field": "request_scheduler",
                          "values": ["serialized_lock", "concurrent_requests", "microbatch"]},
            "execution_adapter": "engine.py and candidate-local Python modules",
            "native_kernel": ("kernels/*.cu|cpp|c plus runtime build outputs"
                              if "native_kernel" in config["optimization"]["allowed_surfaces"]
                              else "not authorized"),
            "rule": "Every candidate still passes source gate, build, runtime precheck, correctness and paired admission.",
        },
        "pending_analyses": [row["iteration"] for row in pending],
        "workload": config["workload"], "protocol": config["evaluation"],
        "loop_protocol": {name: config[name] for name in ("telemetry", "profiling", "probe", "refinement")},
        "workload_roles": config["workload_roles"],
        "workspace": str(workspace), "previous": previous,
        "previous_analysis": previous_analysis,
        "history": history_context,
        "history_total": len(history), "memory_index": "memory/index.json",
        "older_history_summary": older_history_summary(memory_entries, campaign_id),
        "experience_knowledge": experience_ledger(memory_entries, campaign_id),
        "history_retrieval": "The recent 6 records are detailed. The compact ledger covers all older attempts without selecting by the previous hypothesis; search_history is for a targeted fact and read_evidence uses offset/limit.",
        "budget": config["budget"], "agent": config["agent"],
        "model": {"name": config["model"].get("name", "vibeforge-local"),
                  "max_context": runtime.get("max_context", 4096)},
        "source": (workspace / "engine.py").read_text()[:28000],
        "instructions": ("Review only the assigned previous experiment and submit evidence-bound previous_analysis and reflection. Do not propose or edit a candidate."
                         if analysis_only else "Explain previous results, propose one falsifiable hypothesis, edit only the workspace, then submit."),
        "model_features": profile["model"].get("text_config", profile["model"]),
        "hardware": profile["hardware"], "environment": profile["environment"],
        "evidence_root": str(paths.campaign(campaign_id)),
    }
    if state.get("stable_ref"):
        value["candidate_contract"] = {
            "version": 3 if config["agent"].get("mode", "local") == "local" else 1,
            "parent_source": str(paths.release(state["stable_ref"])),
            "observation_paths": [f"iterations/{iteration:04d}/observation.json"],
            "history_iterations": [row["iteration"] for row in history if row["iteration"] < iteration],
            "history_refs": [history_reference(campaign_id, row["iteration"])
                             for row in history if row["iteration"] < iteration],
            "tested_candidates": [{"campaign_id": campaign_id, "iteration": row["iteration"],
                                   "history_ref": history_reference(campaign_id, row["iteration"]),
                                   "fingerprint": row["candidate_gate"]["fingerprint"],
                                   "effect_fingerprint": row.get("effect_fingerprint"),
                                   "effect_scope": row.get("effect_scope")}
                                  for row in history if row.get("parent") == state["stable_ref"]
                                  and row.get("candidate_gate", {}).get("valid")
                                  and row.get("complete")
                                  and reusable_candidate_result(row.get("decision"), row.get("experiment_verification"))],
            "attempted_strategies": [{
                "campaign_id": campaign_id,
                "iteration": row["iteration"],
                "history_ref": history_reference(campaign_id, row["iteration"]),
                "parent": row.get("parent"),
                "decision": row.get("decision"),
                "change": ((row.get("experiment_plan") or {}).get("change") or row.get("hypothesis") or "")[:240],
                **strategy_from_plan(row.get("experiment_plan")),
            } for row in history if row.get("parent") == state["stable_ref"]],
        }
        observation = directory / value["candidate_contract"]["observation_paths"][0]
        if not observation.is_file():
            observation = directory / "observations" / state["stable_ref"] / "baseline.json"
        value["performance_observation"] = read_json(observation) if observation.is_file() else None
        if value["performance_observation"]:
            bounds = observation_benefit_bounds(value["performance_observation"])
            if bounds:
                value["performance_observation"]["benefit_bounds"] = bounds
        value["effective_runtime"] = (value["performance_observation"] or {}).get("runtime", {})
        for entry in ((value["performance_observation"] or {}).get("profile_matrix", {}).get("entries", [])):
            evidence_path = entry.get("evidence_path")
            if isinstance(evidence_path, str):
                value["candidate_contract"]["observation_paths"].append(evidence_path)
        prior_campaigns = ([] if isolated else _database_candidates(
            database, paths, campaign_id, state["stable_ref"], profile_hash, config["config_hash"]))
        if prior_campaigns:
            value["candidate_contract"]["tested_candidates"].extend(prior_campaigns)
            value["candidate_contract"]["attempted_strategies"].extend({
                "campaign_id": row["campaign_id"],
                "iteration": row["iteration"],
                "history_ref": row["history_ref"],
                "parent": state["stable_ref"],
                "decision": row.get("decision"),
                "change": ((row.get("experiment_plan") or {}).get("change") or row.get("hypothesis") or "")[:240],
                **strategy_from_plan(row.get("experiment_plan")),
            } for row in prior_campaigns)
            value["candidate_contract"]["history_refs"].extend(row["history_ref"] for row in prior_campaigns)
            value["archived_candidate_results"] = [
                {key: row[key] for key in (
                    "campaign_id", "iteration", "history_ref", "candidate", "hypothesis", "decision", "verification", "evidence_path")}
                for row in prior_campaigns[-6:]
            ]
    archive_index = directory / "memory" / "archived-campaigns.json"
    if not isolated and archive_index.is_file():
        entries = []
        tested_archives = []
        archived_experience = {"verified_strategies": [], "counterexamples": []}
        for entry in read_json(archive_index).get("entries", []):
            memory_path = confined(directory, entry["memory_index"])
            if not memory_path.is_file():
                continue
            entries.append({key: entry[key] for key in (
                "campaign_id", "config_hash", "profile_hash", "stable_ref", "memory_index",
                "evidence_directory", "completed_attempts", "accepted_candidates") if key in entry})
            tested_archives.extend(archived_candidates(
                directory, entry, state.get("stable_ref"), profile_hash, config["config_hash"]))
            derived = experience_ledger(read_json(memory_path).get("entries", []), entry["campaign_id"])
            for key in archived_experience:
                for item in derived[key]:
                    item["compatibility"] = {
                        name: entry.get(name) for name in
                        ("config_hash", "profile_hash", "stable_ref") if name in entry
                    }
                    evidence = confined(directory, entry["evidence_directory"])
                    item["evidence_path"] = str(
                        (evidence / item["evidence_path"]).relative_to(directory))
                    archived_experience[key].append(item)
        if entries:
            value["archived_campaign_memory"] = {
                "entries": entries[-4:],
                "role": "Historical campaigns, not current measurements or pending analyses. Before selecting a change, read relevant prior memory and patches with read_evidence. The supplied memory_index is root-relative; prefix other paths found inside an archive with its evidence_directory. Only this campaign's bound fresh observations can justify its candidate.",
            }
        if any(archived_experience.values()):
            value["archived_experience_knowledge"] = {
                **archived_experience,
                "scope": "Transfer hints and counterexamples only. Compatibility metadata must be checked; current bound evidence remains mandatory.",
            }
        if tested_archives and value.get("candidate_contract"):
            value["candidate_contract"]["tested_candidates"].extend(tested_archives)
            value["candidate_contract"]["attempted_strategies"].extend({
                "campaign_id": row["campaign_id"],
                "iteration": row["iteration"],
                "history_ref": row["history_ref"],
                "parent": state.get("stable_ref"),
                "decision": row.get("decision"),
                "change": ((row.get("experiment_plan") or {}).get("change")
                           or row.get("hypothesis") or "")[:240],
                **strategy_from_plan(row.get("experiment_plan")),
            } for row in tested_archives)
            value["candidate_contract"]["history_refs"].extend(
                history_reference(row["campaign_id"], row["iteration"]) for row in tested_archives)
            archive_summaries = [
                {key: row[key] for key in ("campaign_id", "iteration", "candidate", "hypothesis", "decision", "verification", "evidence_path")}
                for row in tested_archives[-6:]
            ]
            for row in archive_summaries:
                row["history_ref"] = history_reference(row["campaign_id"], row["iteration"])
            value["archived_candidate_results"] = [
                *value.get("archived_candidate_results", []), *archive_summaries
            ][-6:]
    if value.get("candidate_contract"):
        value["candidate_contract"]["history_refs"] = sorted(set(value["candidate_contract"]["history_refs"]))
    if not analysis_only:
        value["transfer_hints"] = transfer_hints(database, paths, campaign_id, config)
        value = {"optimization_brief": build_optimization_brief(value), **value}
    write_json(paths.campaign(campaign_id) / "resume" / "context.json", value)
    return value
