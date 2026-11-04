import pytest
from vibeforge.storage.atomic import read_json, write_json
from vibeforge.storage.database import Database
from vibeforge.storage.locks import exclusive_lock, LockBusy
from vibeforge.storage.paths import confined, identifier


def test_atomic_json_roundtrip_and_failed_encoding_preserves_old_file(tmp_path):
    target = tmp_path / "state.json"
    write_json(target, {"stable_ref": "v0"})
    with pytest.raises(ValueError):
        write_json(target, {"invalid": float("nan")})
    assert read_json(target) == {"stable_ref": "v0"}


def test_state_and_outbox_rollback_together(tmp_path):
    database = Database(tmp_path / "state.sqlite3")
    database.put("c", {"state": "READY"})
    with pytest.raises(RuntimeError):
        with database.transaction():
            database.put("c", {"state": "BROKEN"})
            database.emit("c", "uncommitted")
            raise RuntimeError("crash before commit")
    assert database.get("c")["state"] == "READY"
    assert list(database.events("c")) == []
    database.close()


def test_lock_prevents_second_supervisor(tmp_path):
    with exclusive_lock(tmp_path / "controller.lock"):
        with pytest.raises(LockBusy):
            with exclusive_lock(tmp_path / "controller.lock"):
                pass


@pytest.mark.parametrize("value", ["../x", "/absolute", ".", "a/b", "a b"])
def test_identifiers_reject_path_components(value):
    with pytest.raises(ValueError):
        identifier(value)


def test_confined_rejects_symlink_escape(tmp_path):
    workspace, other = tmp_path / "workspace", tmp_path / "other"
    workspace.mkdir()
    other.mkdir()
    (workspace / "escape").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError):
        confined(workspace, "escape/output.py")
