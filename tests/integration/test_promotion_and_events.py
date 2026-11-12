import json
import pytest
from vibeforge.observability.events import EventLog
from vibeforge.storage.database import Database
from vibeforge.versioning.promotion import begin_promotion, commit_promotion
from vibeforge.versioning.rollback import cancel_promotion
from vibeforge.controller.recovery import recover
from vibeforge.versioning.candidate import create_workspace
from vibeforge.versioning.release import freeze
from vibeforge.versioning.artifacts import verify_release


def test_promotion_commit_is_idempotent_and_abort_preserves_stable(paths):
    database = Database(paths.data / "state/controller.sqlite3")
    database.put("c", {"state": "READY", "stable_ref": "v0"})
    operation = begin_promotion(database, "c", "v1")
    assert database.get("c")["stable_ref"] == "v0"
    assert commit_promotion(database, "c", operation)["stable_ref"] == "v1"
    commit_promotion(database, "c", operation)
    assert database.get("c")["previous_stable_ref"] == "v0"
    second = begin_promotion(database, "c", "v2")
    cancel_promotion(database, "c", "smoke_failed")
    with pytest.raises(ValueError):
        commit_promotion(database, "c", second)
    assert database.get("c")["stable_ref"] == "v1"
    database.close()


def test_outbox_replays_and_repairs_torn_log(paths):
    database = Database(paths.data / "state/controller.sqlite3")
    database.emit("c", "first")
    log = EventLog(database, "c", paths.campaign("c") / "events")
    log.sync()
    with log.path.open("ab") as stream:
        stream.write(b'{"torn":')
    database.emit("c", "second")
    log.sync()
    log.sync()
    events = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [entry["event_type"] for entry in events] == ["first", "second"]
    database.close()


def test_recovery_aborts_uncommitted_target(paths):
    database = Database(paths.data / "state/controller.sqlite3")
    database.put("c", {"state": "READY", "stable_ref": "v0"})
    begin_promotion(database, "c", "bad")
    database.iteration("c", 1, {"complete": False})
    state = recover(database, paths, "c")
    assert state["stable_ref"] == "v0"
    assert state["pending_operation"] is None
    assert database.iteration("c", 1)["decision"]["reason"] == "interrupted_iteration"
    database.close()


def test_recovery_preserves_terminal_decision_recorded_before_interrupt(paths):
    database = Database(paths.data / "state/controller.sqlite3")
    database.put("terminal", {"state": "ROLLING_BACK", "stable_ref": "v0"})
    database.iteration("terminal", 1, {"complete": False,
        "decision": {"outcome": "reject", "reason": "correctness_failed"}})
    recover(database, paths, "terminal")
    row = database.iteration("terminal", 1)
    assert row["decision"]["reason"] == "correctness_failed"
    assert row["interrupted_after_decision"] and row["interrupted"]
    database.close()


def test_immutable_release_detects_changed_code(paths, project):
    workspace = create_workspace(project / "templates/engine/toy", paths.workspace("c", 0))
    (workspace / 'adapter').mkdir()
    (workspace / 'adapter/manifest.json').write_text('{"kernel":"custom"}')
    manifest = freeze(workspace, paths, None, "python3", "config")
    release = paths.release(manifest["version_id"])
    copied = create_workspace(release, paths.workspace('c', 1))
    assert (copied / 'adapter/manifest.json').read_text() == '{"kernel":"custom"}'
    assert not (copied / 'manifest.json').exists()
    (release / "engine.py").write_text("tampered")
    with pytest.raises(ValueError, match="modified"):
        verify_release(release, manifest)
