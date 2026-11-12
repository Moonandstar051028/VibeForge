"""Rebuildable experimental memory: recorded facts stay separate from interpretation."""

from collections import Counter, defaultdict

from vibeforge.storage.atomic import read_json, write_json
from vibeforge.storage.paths import confined


EXCLUDED_CANDIDATE_REASONS = {
    "execution_failed", "interrupted_iteration", "recovered_incomplete_iteration", "agent_protocol_error",
}


def reusable_candidate_result(decision, verification=None):
    """Keep completed measurements in dedup history, including conclusive invalid gates."""
    decision = decision or {}
    if decision.get("reason") in EXCLUDED_CANDIDATE_REASONS:
        return False
    if decision.get("outcome") in {"accept", "reject"}:
        return True
    verification = verification or {}
    return (decision.get("outcome") == "invalid"
            and isinstance(verification.get("gain"), (int, float))
            and isinstance(verification.get("pairs"), int)
            and verification["pairs"] > 0)


def archived_candidates(directory, entry, stable_ref, profile_hash, config_hash):
    """Recover completed candidates only under the same parent and frozen conditions."""
    if entry.get("config_hash") != config_hash or entry.get("profile_hash") != profile_hash:
        return []
    archive = confined(directory, entry["evidence_directory"])
    index = confined(directory, entry["memory_index"])
    if not index.is_relative_to(archive):
        raise ValueError("Archived memory index must belong to its evidence directory")
    result = []
    for row in read_json(index).get("entries", []):
        facts = row.get("facts") or {}
        gate = facts.get("candidate_gate") or {}
        decision = facts.get("decision") or {}
        verification = facts.get("verification") or {}
        if (row.get("parent") != stable_ref or not row.get("candidate")
                or not facts.get("complete") or not gate.get("valid")
                or not gate.get("fingerprint") or not reusable_candidate_result(decision, verification)):
            continue
        evidence = confined(archive, row["evidence_path"])
        result.append({
            "campaign_id": entry["campaign_id"], "iteration": row["iteration"],
            "history_ref": f"{entry['campaign_id']}:{row['iteration']}",
            "fingerprint": gate["fingerprint"], "candidate": row["candidate"],
            "effect_fingerprint": facts.get("effect_fingerprint"),
            "effect_scope": facts.get("effect_scope"),
            "hypothesis": (row.get("hypothesis") or "")[:240], "decision": decision,
            "experiment_plan": row.get("plan"),
            "verification": verification,
            "evidence_path": str(evidence.relative_to(directory)),
        })
    return result


def memory_entry(row):
    return {
        "iteration": row["iteration"], "experiment_id": row.get("experiment_id"),
        "parent": row.get("parent"), "candidate": row.get("candidate"),
        "hypothesis": row.get("hypothesis"), "plan": row.get("experiment_plan"),
        "facts": {"decision": row.get("decision"), "error": row.get("error"),
                  "verification": row.get("experiment_verification"),
                  "effect_fingerprint": row.get("effect_fingerprint"),
                  "effect_scope": row.get("effect_scope"),
                  "candidate_gate": row.get("candidate_gate"), "complete": row.get("complete", False)},
        "interpretation": row.get("interpretation", "")[:2000],
        "reflection": row.get("reflection"), "analysis_pending": row.get("analysis_pending", False),
        "evidence_path": f"iterations/{row['iteration']:04d}",
    }


def _experience_entry(row, campaign_id):
    facts = row.get("facts") or {}
    decision = facts.get("decision") or {}
    verification = facts.get("verification") or {}
    return {
        "history_ref": f"{campaign_id}:{row['iteration']}",
        "campaign_id": campaign_id, "iteration": row["iteration"],
        "hypothesis": (row.get("hypothesis") or "")[:240],
        "decision": {key: decision.get(key) for key in (
            "outcome", "reason", "objective", "gain", "lower", "upper") if key in decision},
        "prediction_status": verification.get("status"),
        "hypothesis_status": (row.get("reflection") or {}).get("hypothesis_status"),
        "plan": {key: (row.get("plan") or {}).get(key) for key in (
            "measured_bottleneck", "cause", "change", "metric", "expected_change_pct",
            "target_profile_ids") if key in (row.get("plan") or {})},
        "evidence_path": row.get("evidence_path", f"iterations/{row['iteration']:04d}"),
    }


def experience_ledger(entries, campaign_id):
    """Derive conservative reusable strategies and counterexamples from recorded facts."""
    verified, counterexamples = [], []
    for row in entries:
        item = _experience_entry(row, campaign_id)
        decision = (row.get("facts") or {}).get("decision") or {}
        if decision.get("outcome") == "accept" and decision.get("reason") == "measured_gain":
            item["status"] = "local_validated"
            item["scope"] = "this recorded campaign/profile only; transfer requires revalidation"
            verified.append(item)
        elif decision.get("reason") in {
            "correctness_failed", "quality_failed", "ineffective_runtime_change",
            "duplicate_runtime_effect", "performance_regression", "no_significant_gain",
            "profile_regression", "candidate_probe_failed",
        } or str(decision.get("reason", "")).startswith("profile_regression:"):
            item["status"] = "counterexample"
            item["failure"] = decision.get("reason")
            item["scope"] = "recorded failure condition; do not repeat without a materially different mechanism or precondition"
            counterexamples.append(item)
    return {"schema_version": 1, "verified_strategies": verified,
            "counterexamples": counterexamples}


def older_history_summary(entries, campaign_id, recent=6):
    """Summarize every older attempt without pretending that aggregation is raw evidence."""
    older = entries[:-recent] if len(entries) > recent else []
    reasons = Counter()
    outcomes = Counter()
    statuses = Counter()
    mechanisms = defaultdict(list)
    for row in older:
        facts = row.get("facts") or {}
        decision = facts.get("decision") or {}
        reasons[decision.get("reason", "unknown")] += 1
        outcomes[decision.get("outcome", "unknown")] += 1
        statuses[(row.get("reflection") or {}).get("hypothesis_status", "unreviewed")] += 1
        plan = row.get("plan") or {}
        label = (plan.get("change") or row.get("hypothesis") or "unspecified")[:160]
        mechanisms[label].append(f"{campaign_id}:{row['iteration']}")
    return {
        "campaign_id": campaign_id, "attempts": len(older),
        "iteration_range": [older[0]["iteration"], older[-1]["iteration"]] if older else None,
        "outcomes": dict(outcomes), "decision_reasons": dict(reasons),
        "hypothesis_statuses": dict(statuses),
        "strategy_ledger": [
            {"strategy": label, "attempts": len(refs), "history_refs": refs}
            for label, refs in mechanisms.items()
        ],
        "retrieval": "Use search_history with a strategy, bottleneck, outcome or history ref for full records.",
    }


def refresh_memory(database, campaign_id, directory):
    rows = [memory_entry(row) for row in database.iterations(campaign_id)]
    write_json(directory / "memory" / "index.json", {"schema_version": 1, "entries": rows})
    for row in rows:
        path = directory / row["evidence_path"] / "memory.json"
        if not path.is_file() or read_json(path) != row:
            write_json(path, row)
    write_json(directory / "memory" / "experience.json", experience_ledger(rows, campaign_id))
    return rows


def search_memory(directory, query="", limit=5):
    import json
    directory = directory.resolve()
    index = directory / "memory" / "index.json"
    entries = []
    if index.is_file():
        current = read_json(index).get("entries", [])
        campaign_id = directory.name
        entries.extend([{**row, "campaign_id": campaign_id,
                         "history_ref": f"{campaign_id}:{row['iteration']}"}
                        for row in current])
    archive_index = directory / "memory" / "archived-campaigns.json"
    if archive_index.is_file():
        for archive in read_json(archive_index).get("entries", []):
            try:
                memory_path = confined(directory, archive["memory_index"])
                evidence = confined(directory, archive["evidence_directory"])
                if not memory_path.is_file() or not memory_path.is_relative_to(evidence):
                    continue
                for row in read_json(memory_path).get("entries", []):
                    relative = row.get("evidence_path", f"iterations/{row['iteration']:04d}")
                    entries.append({
                        **row, "campaign_id": archive["campaign_id"],
                        "history_ref": f"{archive['campaign_id']}:{row['iteration']}",
                        "evidence_path": str((evidence / relative).relative_to(directory)),
                        "compatibility": {
                            key: archive.get(key) for key in
                            ("config_hash", "profile_hash", "stable_ref") if key in archive
                        },
                    })
            except (KeyError, OSError, TypeError, ValueError):
                continue
    terms = query.casefold().split()
    scored = [(sum(term in json.dumps(row, ensure_ascii=False).casefold() for term in terms), row)
              for row in entries]
    scored = [(score, row) for score, row in scored if score or not terms]
    scored.sort(key=lambda item: (item[0], item[1].get("campaign_id", ""), item[1]["iteration"]), reverse=True)
    return [row for _, row in scored[:max(1, min(int(limit), 10))]]


def reflection_error(previous, reflection, evidence_root):
    """Require an explicit experiment attribution and evidence, not a placeholder."""
    if not previous or not previous.get("analysis_pending"):
        return None
    from pathlib import Path
    from vibeforge.storage.paths import confined
    if not isinstance(reflection, dict) or reflection.get("iteration") != previous["iteration"]:
        return "reflection.iteration must identify the pending experiment"
    if reflection.get("hypothesis_status") not in {"supported", "refuted", "inconclusive"}:
        return "reflection.hypothesis_status must be supported, refuted or inconclusive"
    reason = (previous.get("decision") or {}).get("reason")
    if ((previous.get("decision") or {}).get("outcome") != "accept" or reason != "measured_gain") and reflection["hypothesis_status"] == "supported":
        return "No valid positive performance result supports this hypothesis"
    verification = previous.get("experiment_verification") or {}
    if reflection["hypothesis_status"] == "supported" and verification and verification.get("status") != "supported":
        return "Measured gain does not establish the predicted magnitude of this hypothesis"
    if reason in {"execution_failed", "interrupted_iteration", "recovered_incomplete_iteration", "agent_protocol_error"} and reflection["hypothesis_status"] != "inconclusive":
        return "Infrastructure failure leaves the performance hypothesis inconclusive"
    for key in ("explanation", "next_action"):
        if not isinstance(reflection.get(key), str) or not reflection[key].strip():
            return f"reflection.{key} is required"
    evidence = reflection.get("evidence_paths")
    prefix = f"iterations/{previous['iteration']:04d}/"
    if not isinstance(evidence, list) or not evidence or not any(isinstance(name, str) and name.startswith(prefix) for name in evidence):
        return "reflection must cite evidence belonging to the pending iteration"
    try:
        if any(not confined(Path(evidence_root), name).is_file() for name in evidence):
            return "reflection cites missing evidence"
    except (ValueError, TypeError):
        return "reflection evidence must be confined campaign-relative paths"
    if reason == "correctness_failed":
        correctness_name = f"iterations/{previous['iteration']:04d}/correctness/report.json"
        if correctness_name not in evidence:
            return "correctness_failed reflection must cite its correctness/report.json"
        report = read_json(confined(Path(evidence_root), correctness_name))
        failed = [check for check in report.get("checks", []) if not check.get("passed")]
        numerical_failed = any(
            isinstance(check.get(section), dict) and not check[section].get("passed", False)
            for check in failed for section in ("numerical", "continuation_numerical")
        )
        explanation = " ".join((reflection.get("explanation", ""), reflection.get("next_action", ""))).casefold()
        unsupported_numerical_claims = (
            "max_normalized_rms", "normalized_rms threshold", "rms threshold",
            "logit divergence", "logit error", "numerical threshold", "numerical tolerance",
        )
        if not numerical_failed and any(term in explanation for term in unsupported_numerical_claims):
            return ("Correctness report shows all recorded numerical/logit checks passed; describe the actual "
                    "generation, repeat, isolation or operational failures instead")
    return None
