"""Rollback a failed activation without losing the last committed stable ref."""

import json


def cancel_promotion(database, campaign_id, reason):
    with database.transaction():
        state = database.get(campaign_id)
        operation = state.get("pending_operation")
        if operation:
            row = database.connection.execute("SELECT status FROM operations WHERE id=?", (operation,)).fetchone()
            if row and row["status"] == "pending":
                database.connection.execute("UPDATE operations SET status='aborted' WHERE id=?", (operation,))
        state.update(pending_operation=None, active_ref=None, state="ROLLING_BACK")
        database.put(campaign_id, state)
        database.emit(campaign_id, "rollback_intent", operation_id=operation,
                      payload={"target": state.get("stable_ref"), "reason": reason})
    return state.get("stable_ref")


def rollback_completed(database, campaign_id, target):
    return database.update(campaign_id, {"active_ref": target, "state": "READY"},
                           "rollback_completed", payload={"target": target})
