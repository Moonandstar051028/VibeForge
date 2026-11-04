import sys
import pytest

from vibeforge.controller.supervisor import Supervisor
from vibeforge.storage.atomic import read_json, write_json
from vibeforge.versioning.candidate import create_workspace
from vibeforge.versioning.release import freeze


def test_bounded_build_freezes_runtime_artifacts(paths, project, toy_config):
    controller = Supervisor(paths, "build-test")
    try:
        controller.config = toy_config
        workspace = create_workspace(project / "templates/engine/toy", paths.workspace("build-test", 0))
        toy_config["engine"]["build_command"] = ["{python}", "-c",
            "from pathlib import Path; Path('runtime').mkdir(); Path('runtime/kernel.bin').write_bytes(b'compiled')"]
        controller.build_workspace(workspace, paths.campaign("build-test"))
        release = freeze(workspace, paths, None, sys.executable, toy_config["config_hash"])
        assert "runtime/kernel.bin" in release["files"]
        assert (paths.release(release["version_id"]) / "runtime/kernel.bin").read_bytes() == b"compiled"
    finally:
        controller.close()


def test_build_cannot_write_controller_state(paths, project, toy_config):
    controller = Supervisor(paths, "build-denied")
    try:
        controller.config = toy_config
        workspace = create_workspace(project / "templates/engine/toy", paths.workspace("build-denied", 0))
        forbidden = paths.data / "state" / "forbidden.txt"
        toy_config["engine"]["build_command"] = ["{python}", "-c",
            f"from pathlib import Path; Path({str(forbidden)!r}).write_text('bad')"]
        with pytest.raises(RuntimeError, match="build failed"):
            controller.build_workspace(workspace, paths.campaign("build-denied"))
        assert not forbidden.exists()
    finally:
        controller.close()


def test_initial_build_cannot_override_campaign_precision(paths, toy_config, monkeypatch):
    controller = Supervisor(paths, "bootstrap-precision")
    try:
        controller.create(toy_config)
        controller.config = toy_config
        def wrong_build(workspace, folder):
            config = read_json(workspace / "engine.json")
            config["dtype"] = "float16"
            write_json(workspace / "engine.json", config)
        monkeypatch.setattr(controller, "build_workspace", wrong_build)
        with pytest.raises(ValueError, match="disagrees"):
            controller.bootstrap()
        assert controller.engine is None
    finally:
        controller.close()
