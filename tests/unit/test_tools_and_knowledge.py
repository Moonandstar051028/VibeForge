import pytest
from vibeforge.agent.runner import parse_object
from vibeforge.agent.tools import AgentTools
from vibeforge.knowledge.validator import validate_rule
from vibeforge.knowledge.retriever import retrieve


def test_tool_writes_are_scoped_and_audited(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    tools = AgentTools(workspace, tmp_path / "commands.jsonl")
    tools.execute("write_file", {"path": "engine.py", "content": "value = 1\n"})
    tools.execute("replace_text", {"path": "engine.py", "old": "1", "new": "2"})
    assert tools.execute("read_file", {"path": "engine.py"}) == "value = 2\n"
    for path in ("../outside.py", "/tmp/outside.py"):
        with pytest.raises(ValueError):
            tools.execute("write_file", {"path": path, "content": "bad"})
    assert len((tmp_path / "commands.jsonl").read_text().splitlines()) == 5


def test_ambiguous_edit_is_rejected(tmp_path):
    (tmp_path / "x.py").write_text("a a")
    tools = AgentTools(tmp_path, tmp_path / "commands.jsonl")
    with pytest.raises(ValueError, match="exactly once"):
        tools.execute("replace_text", {"path": "x.py", "old": "a", "new": "b"})


def test_json_protocol_handles_thinking_and_prefix_text():
    assert parse_object('<think>reasoning</think>\njson:\n{"tool":"read_file"}')["tool"] == "read_file"
    with pytest.raises(ValueError):
        parse_object("No structured output")


def test_knowledge_does_not_generalize_single_profile():
    rule = {"rule_id": "r", "status": "cross_config_validated", "conditions": {"device": "thor"},
            "implementation": "code", "validation": "measured", "evidence": [{"profile_hash": "a"}]}
    with pytest.raises(ValueError, match="distinct"):
        validate_rule(rule)
    rule["status"] = "local_validated"
    assert retrieve([rule], {"device": "other"}) == []
    assert retrieve([rule], {"device": "thor"}) == [rule]
    rule["status"] = "draft"
    assert retrieve([rule], {"device": "thor"}) == []
