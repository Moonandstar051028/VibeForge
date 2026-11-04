"""Resolve pending transactions and interrupted rounds before starting an agent."""

from vibeforge.engine.process import stop_record
from vibeforge.versioning.rollback import cancel_promotion


def recover(database, paths, campaign_id):
    state = database.get(campaign_id)
    stop_record(paths.data / "state" / f"{campaign_id}.agent.json")
    stop_record(paths.data / "state" / f"{campaign_id}.engine.json")
    if state.get("pending_operation"):
        cancel_promotion(database, campaign_id, "supervisor_restart_before_commit")
    with database.transaction():
        state = database.get(campaign_id)
        for iteration in database.iterations(campaign_id):
            if not iteration.get("complete"):
                iteration.update(complete=True, interrupted=True, analysis_pending=True,
                                 decision={"outcome": "reject", "reason": "interrupted_iteration"})
                database.iteration(campaign_id, iteration["iteration"], iteration)
        state.update(active_ref=None, state="RECOVERING",
                     completed_iterations=len(database.iterations(campaign_id)))
        database.put(campaign_id, state)
        database.emit(campaign_id, "recovery_reconciled", payload={"stable_ref": state.get("stable_ref")})
    return state
