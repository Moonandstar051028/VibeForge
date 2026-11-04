import os
import subprocess
import sys
from vibeforge.storage.atomic import write_json
from vibeforge.storage.database import Database
from vibeforge.engine.process import stop_record


def test_real_supervisor_crash_during_promotion_recovers_from_disk(paths, project, toy_config):
    config = paths.data / "crash-campaign.json"
    write_json(config, toy_config)
    command = [sys.executable, "-m", "vibeforge", "--project", str(project),
               "--data-root", str(paths.data)]
    env = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONDONTWRITEBYTECODE="1",
               VIBEFORGE_FAILPOINT="after_promotion_intent")
    try:
        first = subprocess.run(command + ["run", "crash", "--config", str(config)],
                               env=env, capture_output=True, text=True, timeout=90)
        assert first.returncode == 86, first.stdout + first.stderr
        database = Database(paths.data / "state/controller.sqlite3")
        interrupted = database.get("crash")
        assert interrupted["pending_operation"]
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
        database.close()
    finally:
        stop_record(paths.data / "state/crash.engine.json")
        stop_record(paths.data / "state/crash.agent.json")
