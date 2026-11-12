"""Resolve pending transactions and interrupted rounds before starting an agent."""

import json
from vibeforge.engine.process import stop_record
from vibeforge.storage.atomic import write_json
from vibeforge.agent.analysis import write_analysis
from vibeforge.versioning.rollback import cancel_promotion


def recover(database, paths, campaign_id):
    state = database.get(campaign_id)
    reconciled = []
    stop_record(paths.data / "state" / f"{campaign_id}.agent.json")
    stop_record(paths.data / "state" / f"{campaign_id}.engine.json")
    stop_record(paths.data / "state" / f"{campaign_id}.build.json")
    stop_record(paths.data / "state" / f"{campaign_id}.quality.json")
    if state.get("pending_operation"):
        cancel_promotion(database, campaign_id, "supervisor_restart_before_commit")
    with database.transaction():
        state = database.get(campaign_id)
        for iteration in database.iterations(campaign_id):
            if not iteration.get("complete"):
                committed = any(json.loads(row["data"])["target"] == iteration.get("candidate")
                                for row in database.connection.execute(
                                    "SELECT data FROM operations WHERE campaign_id=? AND kind='promotion' AND status='committed'",
                                    (campaign_id,)))
                recorded = iteration.get("decision") or {}
                terminal = recorded.get("reason") and recorded.get("reason") != "unfinished"
                decision = (recorded if terminal else
                            {"outcome": "accept", "reason": "promotion_committed"} if committed else
                            {"outcome": "reject", "reason": "interrupted_iteration"})
                iteration.update(complete=True, interrupted=True, analysis_pending=True, decision=decision)
                if terminal:
                    iteration["interrupted_after_decision"] = True
                database.iteration(campaign_id, iteration["iteration"], iteration)
                reconciled.append(iteration["iteration"])
        iterations = database.iterations(campaign_id)
        state.update(active_ref=None, state="RECOVERING",
                     analysis_pending=any(row.get("analysis_pending") for row in iterations),
                     completed_iterations=max((row["iteration"] for row in iterations), default=0))
        # Legacy iteration IDs remain attempt IDs; never reuse an interrupted folder.
        state["proposal_attempts"] = state["completed_iterations"]
        state["valid_candidates"] = sum(bool(row.get("experiment_id") or row.get("candidate")) for row in iterations)
        state["completed_experiments"] = sum(bool(row.get("experiment_id") or row.get("candidate"))
                                             and row.get("complete", False)
                                             and (row.get("decision") or {}).get("reason") != "interrupted_iteration"
                                             for row in iterations)
        state["stalled_attempts"] = 0
        database.put(campaign_id, state)
        database.emit(campaign_id, "recovery_reconciled", payload={"stable_ref": state.get("stable_ref")})
    for number in reconciled:
        iteration = database.iteration(campaign_id, number)
        folder = paths.campaign(campaign_id) / "iterations" / f"{number:04d}"
        write_json(folder / "decision.json", iteration["decision"])
        write_analysis(folder / "analysis.md", iteration, None)
    return state
