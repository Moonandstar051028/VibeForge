import time
import pytest
from vibeforge.contracts.campaign import load_config
from vibeforge.controller.budgets import stop_reason
from vibeforge.controller.state_machine import transition
from vibeforge.storage.atomic import write_json
from vibeforge.storage.database import Database
from vibeforge.storage.atomic import digest, read_json
from vibeforge.controller.supervisor import Supervisor


def test_protocol_records_concurrency_and_rejects_out_of_range(project, tmp_path, toy_config):
    toy_config["workload"]["concurrency"] = 8
    path = tmp_path / "campaign.json"
    write_json(path, toy_config)
    assert load_config(path, project)["evaluation"]["concurrency"] == 8
    toy_config["workload"]["concurrency"] = 65
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="concurrency"):
        load_config(path, project)


def test_resource_sampling_protocol_is_frozen_and_validated(project, tmp_path, toy_config):
    toy_config["evaluation"].update(
        device_sample_interval=0.25, min_tail_requests=200,
        require_continuous_device_telemetry=False)
    path = tmp_path / "campaign.json"
    write_json(path, toy_config)

    protocol = load_config(path, project)["evaluation"]

    assert protocol["device_sample_interval"] == 0.25
    assert protocol["min_tail_requests"] == 200
    toy_config["evaluation"]["device_sample_interval"] = 0
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="device_sample_interval"):
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


def test_thinking_configuration_is_validated_and_model_scoped(project, tmp_path, toy_config):
    toy_config["agent"].update(mode="local", framework="qwen-agent", max_new_tokens=512)
    toy_config["agent"]["thinking"] = {"enabled": True, "temperature": 0.6,
                                         "top_p": 0.95, "top_k": 20, "min_p": 0.0}
    path = tmp_path / "campaign.json"
    write_json(path, toy_config)

    config = load_config(path, project)

    assert config["agent"]["thinking"]["enabled"] is True
    assert config["agent"]["max_new_tokens"] < config["model"]["max_context"]

    toy_config["agent"]["thinking"]["temperature"] = 0
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="thinking.temperature"):
        load_config(path, project)


def test_codex_agent_is_selectable_and_codex_owns_compaction(project, tmp_path, toy_config):
    toy_config["agent"] = "configs/agent/codex.json"
    path = tmp_path / "campaign.json"
    write_json(path, toy_config)
    referenced = load_config(path, project)
    assert referenced["agent"]["framework"] == "codex"
    assert referenced["agent"]["model"] == referenced["model"]["name"]

    toy_config["agent"] = {
        "mode": "local", "framework": "codex", "prompt": "templates/prompts/codex-implementation.md",
        "max_turns": 6, "max_new_tokens": 512, "executable": "codex",
    }
    write_json(path, toy_config)
    config = load_config(path, project)
    assert config["agent"]["framework"] == "codex"
    assert "compact_at_tokens" not in config["agent"]
    assert config["agent"]["model"] == config["model"]["name"]

    toy_config["agent"]["prompt"] = "templates/prompts/implementation.md"
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="Codex-specific prompt"):
        load_config(path, project)
    toy_config["agent"]["prompt"] = "templates/prompts/codex-implementation.md"

    toy_config["agent"]["compact_at_tokens"] = 4096
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="Codex owns"):
        load_config(path, project)


def test_context_window_cannot_exceed_model_native_limit(project, tmp_path, toy_config):
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "config.json", {"model_type": "fixture", "max_position_embeddings": 8192})
    toy_config["model"] = {"name": "bounded", "path": str(model), "device": "cpu",
                           "fixture": True, "max_context": 16384}
    path = tmp_path / "campaign.json"
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="native limit"):
        load_config(path, project)

    toy_config["model"]["max_context"] = 8192
    write_json(path, toy_config)
    assert load_config(path, project)["model"]["native_max_context"] == 8192


def test_context_window_defaults_to_native_model_limit(project, tmp_path, toy_config):
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "config.json", {
        "model_type": "fixture", "text_config": {"max_position_embeddings": 262144},
    })
    toy_config["model"] = {"name": "auto-window", "path": str(model), "device": "cpu", "fixture": True}
    path = tmp_path / "campaign.json"
    write_json(path, toy_config)

    config = load_config(path, project)

    assert config["model"]["max_context"] == 262144
    assert config["model"]["native_max_context"] == 262144
    assert config["model"]["context_limit_source"] == "model_config"


def test_invalid_state_jump_preserves_state(tmp_path):
    database = Database(tmp_path / "state.sqlite3")
    database.put("c", {"state": "READY"})
    with pytest.raises(ValueError):
        transition(database, "c", "COMMITTING")
    assert database.get("c")["state"] == "READY"
    database.close()


def test_schema_v4_freezes_loop_protocol_and_workload_roles(project, tmp_path, toy_config):
    legacy = {key: value for key, value in toy_config.items()
              if key not in {"profiling", "telemetry", "probe", "refinement", "workload_roles",
                             "schema_version", "config_hash"}}
    legacy["observation"] = {"max_new_tokens": 24, "max_profiles_per_session": 1,
                             "max_age_seconds": 90}
    path = tmp_path / "campaign.json"
    write_json(path, legacy)

    config = load_config(path, project)

    assert config["schema_version"] == 4
    assert "observation" not in config
    assert config["profiling"]["max_new_tokens"] == 24
    assert config["telemetry"]["capture_agent_requests"] is True
    assert config["probe"]["enabled"] is False
    assert config["refinement"]["enabled"] is False
    assert config["workload_roles"] == {
        "deployment": {"profile_ids": ["default"], "admission": True},
        "agent_self_hosting": {
            "enabled": True, "source": "live_agent_requests", "admission": False,
        },
    }


def test_loop_protocol_rejects_unsafe_or_inconsistent_settings(project, tmp_path, toy_config):
    path = tmp_path / "campaign.json"
    toy_config.pop("config_hash", None)
    toy_config["refinement"].update(enabled=True, max_revisions=1)
    toy_config["probe"]["enabled"] = False
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="requires probe.enabled"):
        load_config(path, project)

    toy_config["probe"]["enabled"] = True
    toy_config["telemetry"].update(enabled=False, capture_agent_requests=True)
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="requires telemetry.enabled"):
        load_config(path, project)

    toy_config["telemetry"].update(enabled=True, capture_agent_requests=True)
    toy_config["probe"]["max_attempts"] = 1
    write_json(path, toy_config)
    with pytest.raises(ValueError, match="initial probe"):
        load_config(path, project)


def test_native_surface_requires_frozen_build_recipe(project, tmp_path, toy_config):
    toy_config.pop("config_hash", None)
    toy_config["optimization"]["allowed_surfaces"] = ["execution_adapter", "native_kernel"]
    path = tmp_path / "native.json"
    write_json(path, toy_config)

    with pytest.raises(ValueError, match="frozen engine.build_command"):
        load_config(path, project)


def test_probe_and_refinement_transitions_are_explicit(tmp_path):
    database = Database(tmp_path / "loop-state.sqlite3")
    try:
        database.put("c", {"state": "BUILDING"})
        transition(database, "c", "PROBING")
        transition(database, "c", "REFINING")
        transition(database, "c", "BUILDING")
        transition(database, "c", "CANDIDATE_FROZEN")
        assert database.get("c")["state"] == "CANDIDATE_FROZEN"
        with pytest.raises(ValueError):
            transition(database, "c", "REFINING")
    finally:
        database.close()


def test_budget_survives_process_restart():
    state = {"started_at": time.time() - 10, "completed_iterations": 0}
    assert stop_reason(state, {"max_seconds": 5, "max_iterations": 2, "max_failures": 3}) == "max_seconds"


def test_candidate_failure_budget_stops_repeated_invalid_experiments():
    state = {"started_at": time.time(), "completed_iterations": 1,
             "consecutive_candidate_failures": 5}
    budget = {"max_seconds": 60, "max_iterations": 20, "max_failures": 3,
              "max_candidate_failures": 5}
    assert stop_reason(state, budget) == "max_candidate_failures"


@pytest.mark.parametrize("agent_output,workload_output,expected", [(2048, 16, 2048), (256, 4096, 4096)])
def test_bootstrap_release_can_serve_agent_and_workload_budgets(
        paths, toy_config, monkeypatch, agent_output, workload_output, expected):
    toy_config["agent"]["max_new_tokens"] = agent_output
    toy_config["workload"]["max_new_tokens"] = workload_output
    toy_config["config_hash"] = digest({k: v for k, v in toy_config.items() if k != "config_hash"})
    controller = Supervisor(paths, "output-budgets")
    observed = {}

    def inspect_release(version, log):
        observed.update(read_json(paths.release(version) / "engine.json"))
        raise RuntimeError("stop before inference")

    monkeypatch.setattr(controller, "start_engine", inspect_release)
    try:
        controller.create(toy_config)
        controller.config = toy_config
        with pytest.raises(RuntimeError, match="stop before inference"):
            controller.bootstrap()
        assert observed["max_new_tokens"] == expected
        assert observed["max_context"] == 32768
    finally:
        controller.close()
