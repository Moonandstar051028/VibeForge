"""Resolve pending transactions and interrupted rounds before starting an agent."""

import json
from vibeforge.engine.process import stop_record
from vibeforge.versioning.rollback import cancel_promotion


def recover(database, paths, campaign_id):
    state = database.get(campaign_id)
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
                decision = (iteration.get("decision") or {"outcome": "accept", "reason": "promotion_committed"}) if committed else {
                    "outcome": "reject", "reason": "interrupted_iteration"}
                iteration.update(complete=True, interrupted=True, analysis_pending=True,
                                 decision=decision)
                database.iteration(campaign_id, iteration["iteration"], iteration)
        iterations = database.iterations(campaign_id)
        state.update(active_ref=None, state="RECOVERING",
                     analysis_pending=any(row.get("analysis_pending") for row in iterations),
                     completed_iterations=max((row["iteration"] for row in iterations), default=0))
        database.put(campaign_id, state)
        database.emit(campaign_id, "recovery_reconciled", payload={"stable_ref": state.get("stable_ref")})
    return state
