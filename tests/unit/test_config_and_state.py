import time
import pytest
from vibeforge.contracts.campaign import load_config
from vibeforge.controller.budgets import stop_reason
from vibeforge.controller.state_machine import transition
from vibeforge.storage.atomic import write_json
from vibeforge.storage.database import Database


def test_protocol_cannot_silently_enable_unsupported_concurrency(project, tmp_path, toy_config):
    toy_config["workload"]["concurrency"] = 8
    path = tmp_path / "campaign.json"
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="single-request"):
        load_config(path, project)


def test_model_and_runtime_paths_are_portable_overrides(project, monkeypatch):
    monkeypatch.setenv("VIBEFORGE_MODEL_PATH", "/models/qwen")
    monkeypatch.setenv("VIBEFORGE_RUNTIME_PYTHON", "/venv/bin/python")
    config = load_config(project / "configs/campaigns/qwen36-35b.json", project)
    assert config["model"]["path"] == "/models/qwen"
    assert config["runtime_python"] == "/venv/bin/python"


def test_relative_model_path_resolves_from_project(project):
    config = load_config(project / "configs/campaigns/qwen36-35b.json", project)
    assert config["model"]["path"] == str((project / "weights/Qwen3.6-35B-A3B").resolve())


def test_local_agent_requires_qwen_agent_and_bounded_call_budget(project, tmp_path, toy_config):
    toy_config["agent"].update(mode="local", framework="custom", max_turns=6)
    path = tmp_path / "campaign.json"
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="agent.framework"):
        load_config(path, project)

    toy_config["agent"].update(framework="qwen-agent", max_turns=21)
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="agent.max_turns"):
        load_config(path, project)


def test_invalid_state_jump_preserves_state(tmp_path):
    database = Database(tmp_path / "state.sqlite3")
    database.put("c", {"state": "READY"})
    with pytest.raises(ValueError):
        transition(database, "c", "COMMITTING")
    assert database.get("c")["state"] == "READY"
    database.close()


def test_budget_survives_process_restart():
    state = {"started_at": time.time() - 10, "completed_iterations": 0}
    assert stop_reason(state, {"max_seconds": 5, "max_iterations": 2, "max_failures": 3}) == "max_seconds"
