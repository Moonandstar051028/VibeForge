"""Promotion intentions and stable pointers are committed transactionally."""

import json
import uuid
from vibeforge.storage.atomic import canonical_json


def begin_promotion(database, campaign_id, target):
    state = database.get(campaign_id)
    operation = uuid.uuid4().hex
    payload = {"source": state.get("stable_ref"), "target": target}
    with database.transaction():
        database.connection.execute("INSERT INTO operations VALUES (?,?,?,?,?)",
                                    (operation, campaign_id, "promotion", "pending", canonical_json(payload)))
        state.update(pending_operation=operation, active_ref=None, state="ACTIVATING")
        database.put(campaign_id, state)
        database.emit(campaign_id, "promotion_intent", operation_id=operation, payload=payload)
    return operation


def commit_promotion(database, campaign_id, operation):
    with database.transaction():
        row = database.connection.execute("SELECT * FROM operations WHERE id=?", (operation,)).fetchone()
        if not row or row["campaign_id"] != campaign_id or row["kind"] != "promotion":
            raise ValueError("Invalid promotion operation")
        if row["status"] == "committed":
            return database.get(campaign_id)
        if row["status"] != "pending":
            raise ValueError("Promotion is no longer pending")
        payload, state = json.loads(row["data"]), database.get(campaign_id)
        if state.get("stable_ref") != payload["source"]:
            raise ValueError("Stable version changed during promotion")
        state.update(stable_ref=payload["target"], previous_stable_ref=payload["source"],
                     active_ref=payload["target"], pending_operation=None, state="READY")
        database.put(campaign_id, state)
        database.connection.execute("UPDATE operations SET status='committed' WHERE id=?", (operation,))
        database.emit(campaign_id, "promotion_committed", operation_id=operation, payload=payload)
    return state
