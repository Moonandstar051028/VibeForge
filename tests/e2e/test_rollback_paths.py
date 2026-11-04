import pytest
from vibeforge.controller.supervisor import Supervisor
from vibeforge.storage.atomic import digest


@pytest.mark.parametrize("action,expected_reason", [
    ({"tool": "replace_text", "arguments": {
        "path": "engine.py", "old": "work = sum(range(120000))",
        "new": "work = sum(range(120000)) + sum(range(120000)) - sum(range(120000))"}},
     "performance_regression"),
    ({"tool": "replace_text", "arguments": {
        "path": "engine.py", "old": "work = sum(range(120000))", "new": "work = ("}},
     "execution_failed"),
    ({"tool": "write_file", "arguments": {"path": "../outside.py", "content": "bad"}},
     "execution_failed"),
])
def test_regression_build_failure_and_agent_failure_restore_stable(paths, toy_config, action, expected_reason):
    toy_config["budget"]["max_iterations"] = 1
    toy_config["agent"]["actions"] = [{"hypothesis": "Controlled failure experiment", "edits": [action]}]
    toy_config["config_hash"] = digest({k: v for k, v in toy_config.items() if k != "config_hash"})
    controller = Supervisor(paths, "failure-path")
    try:
        controller.create(toy_config)
        controller.run()
        state = controller.database.get("failure-path")
        decision = controller.database.iteration("failure-path", 1)["decision"]
        assert state["state"] == "STOPPED"
        assert decision["reason"] == expected_reason
        assert state["stable_ref"] == state["initial_ref"]
        assert state["analysis_pending"] is False
    finally:
        controller.close()
