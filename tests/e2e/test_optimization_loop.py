from vibeforge.controller.supervisor import Supervisor
from vibeforge.storage.atomic import digest, read_json


def test_two_rounds_accept_good_change_reject_wrong_change_and_resume(paths, toy_config):
    controller = Supervisor(paths, "closed-loop")
    try:
        controller.create(toy_config)
        controller.run()
        state = controller.database.get("closed-loop")
        iterations = controller.database.iterations("closed-loop")
        assert state["state"] == "STOPPED"
        assert state["analysis_pending"] is False
        assert iterations[0]["decision"]["outcome"] == "accept"
        assert iterations[1]["decision"]["reason"] == "correctness_failed"
        assert state["stable_ref"] == iterations[0]["candidate"]
        assert state["stable_ref"] != iterations[1]["candidate"]
        assert state["previous_stable_ref"] == state["initial_ref"]
        controller.run(resume=True)
        assert controller.database.get("closed-loop")["completed_iterations"] == 2
        events = list(controller.database.events("closed-loop"))
        sessions = [e for e in events if e["event_type"] == "agent_completed"]
        assert len({e["payload"]["session_id"] for e in sessions}) == 3
        assert (paths.campaign("closed-loop") / "summary.md").exists()
    finally:
        controller.close()


def test_controller_starts_qwen_agent_worker_and_final_analysis(paths, toy_config):
    toy_config["agent"] = {
        "mode": "local", "framework": "qwen-agent", "max_turns": 4,
        "max_new_tokens": 256, "timeout_seconds": 20, "request_timeout": 10,
        "prompt": "templates/prompts/implementation.md",
    }
    toy_config["budget"]["max_iterations"] = 1
    toy_config["evaluation"]["max_retests"] = 0
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items() if key != "config_hash"})
    controller = Supervisor(paths, "qwen-agent-worker")
    try:
        controller.create(toy_config)
        state = controller.run()
        assert state["state"] == "STOPPED" and not state["analysis_pending"]
        directory = paths.campaign("qwen-agent-worker")
        for path in (directory / "iterations/0001/agent-result.json", directory / "final-analysis/agent-result.json"):
            result = read_json(path)
            assert result["mode"] == "qwen_agent" and result["complete"]
        assert read_json(directory / "bootstrap/agent-smoke.json")["framework"] == "Qwen-Agent"
        assert (directory / "bootstrap/agent-smoke.jsonl").exists()
        assert not (paths.data / "state/qwen-agent-worker.agent.json").exists()
    finally:
        controller.close()
