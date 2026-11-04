import os
import subprocess
import sys
import pytest
from vibeforge.storage.atomic import write_json
from vibeforge.storage.database import Database
from vibeforge.engine.process import stop_record


@pytest.mark.parametrize("failpoint,exit_code,committed", [
    ("after_promotion_intent", 86, False), ("after_promotion_commit", 87, True)])
def test_real_supervisor_crash_during_promotion_recovers_from_disk(paths, project, toy_config, failpoint, exit_code, committed):
    config = paths.data / "crash-campaign.json"
    write_json(config, toy_config)
    command = [sys.executable, "-m", "vibeforge", "--project", str(project),
               "--data-root", str(paths.data)]
    env = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONDONTWRITEBYTECODE="1",
               VIBEFORGE_FAILPOINT=failpoint)
    try:
        first = subprocess.run(command + ["run", "crash", "--config", str(config)],
                               env=env, capture_output=True, text=True, timeout=90)
        assert first.returncode == exit_code, first.stdout + first.stderr
        database = Database(paths.data / "state/controller.sqlite3")
        interrupted = database.get("crash")
        assert bool(interrupted["pending_operation"]) is not committed
        initial = interrupted["stable_ref"]
        database.close()
        env.pop("VIBEFORGE_FAILPOINT")
        resumed = subprocess.run(command + ["resume", "crash"], env=env,
                                 capture_output=True, text=True, timeout=90)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        database = Database(paths.data / "state/controller.sqlite3")
        state = database.get("crash")
        assert state["state"] == "STOPPED"
        assert state["pending_operation"] is None
        assert state["stable_ref"] == initial
        assert database.iteration("crash", 1)["interrupted"]
        assert database.iteration("crash", 1)["decision"]["outcome"] == ("accept" if committed else "reject")
        assert not state["analysis_pending"]
        database.close()
    finally:
        stop_record(paths.data / "state/crash.engine.json")
        stop_record(paths.data / "state/crash.agent.json")
