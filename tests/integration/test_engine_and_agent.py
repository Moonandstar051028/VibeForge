import json
import sys
from vibeforge.agent.runner import run_agent, smoke_agent
from vibeforge.engine.process import EngineProcess
from vibeforge.evaluation.reference import capture_reference
from vibeforge.evaluation.correctness import evaluate_correctness


def test_engine_http_reference_and_json_protocol(paths, project):
    engine = EngineProcess(paths, "transport", sys.executable, startup_timeout=10)
    try:
        client = engine.start(project / "templates/engine/toy", paths.campaign("transport") / "engine.log")
        assert smoke_agent(client)["passed"]
        reference = capture_reference(client, [{"id": "x", "prompt": "hello", "max_new_tokens": 2}],
                                      paths.campaign("transport") / "reference.json", {"fixture": True})
        assert evaluate_correctness(client, reference, {"atol": 0, "rtol": 0})["passed"]
    finally:
        engine.stop()
    assert not engine.record.exists()


def test_local_agent_uses_model_actions_to_edit_and_submit(tmp_path, project):
    class LocalClient:
        replies = iter([
            {"tool": "replace_text", "arguments": {"path": "engine.py", "old": "1", "new": "2"}},
            {"tool": "check_python", "arguments": {"path": "engine.py"}},
            {"tool": "submit_candidate", "hypothesis": "missing prior analysis"},
            {"tool": "submit_candidate", "hypothesis": "equivalent optimization", "previous_analysis": "initial"},
        ])

        def chat(self, **arguments):
            return {"text": json.dumps(next(self.replies))}

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "engine.py").write_text("value = 1\n")
    result = run_agent(LocalClient(), {"agent": {"mode": "local", "max_turns": 4},
                                      "previous": {"analysis_pending": True}}, workspace,
                       tmp_path / "trace.jsonl", project / "templates/prompts/implementation.md")
    assert result["mode"] == "local" and result["complete"]
    assert (workspace / "engine.py").read_text() == "value = 2\n"
