from pathlib import Path

import vibeforge.engine.process as engine_process
from vibeforge.engine.process import EngineProcess
from vibeforge.storage.paths import Paths


def test_managed_engine_disables_inductor_process_pool(tmp_path, monkeypatch):
    paths = Paths.create(tmp_path / "project")
    captured = {}

    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    class FakeClient:
        def __init__(self, url, timeout):
            self.url = url

        def health(self):
            return {"instance_token": captured["command"][-1]}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(engine_process, "stop_record", lambda _: False)
    monkeypatch.setattr(engine_process, "process_identity", lambda pid: {
        "pid": pid, "start_ticks": "1", "boot_id": "test", "state": "S"
    })
    monkeypatch.setattr(engine_process, "write_json", lambda *_: None)
    monkeypatch.setattr(engine_process.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(engine_process, "EngineClient", FakeClient)

    runner = EngineProcess(paths, "test-campaign", "python")
    runner.start(tmp_path / "release", tmp_path / "logs" / "engine.log")

    assert captured["env"]["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
