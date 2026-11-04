import importlib.util
import json

import pytest

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


def test_transformers_backend_parses_qwen_native_tool_markup(project):
    path = project / "templates/engine/transformers/engine.py"
    spec = importlib.util.spec_from_file_location("transformer_template", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = (
        "<tool_call>\n<function=replace_text>\n<parameter=path>\nengine.py\n</parameter>\n"
        "<parameter=old>\nvalue = 1\n</parameter>\n<parameter=new>\nvalue = 2\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert module.parse_tool_calls(output) == [{
        "name": "replace_text",
        "arguments": {"path": "engine.py", "old": "value = 1", "new": "value = 2"},
    }]


def test_transformers_chat_completion_uses_native_tools(project, monkeypatch):
    path = project / "templates/engine/transformers/engine.py"
    spec = importlib.util.spec_from_file_location("transformer_template", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tools"][0]["function"]["name"] == "read_file"
            assert kwargs["enable_thinking"] is False
            assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"path": "engine.py"}
            return "rendered prompt"

    monkeypatch.setattr(module, "TOKENIZER", Tokenizer())
    monkeypatch.setattr(module, "generate", lambda prompt, **kwargs: {
        "text": "<tool_call><function=read_file><parameter=path>engine.py</parameter></function></tool_call>",
        "input_ids": [1, 2], "token_ids": [3],
    })
    result = module.chat_completion(
        [{"role": "system", "content": "help"}, {"role": "user", "content": "inspect"},
         {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
             "name": "read_file", "arguments": '{"path":"engine.py"}'}}]},
         {"role": "tool", "content": "value = 1"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        max_tokens=32,
    )
    assert json.loads(result["tool_calls"][0]["arguments"]) == {"path": "engine.py"}
    assert result["usage"] == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}


def test_native_tool_parser_preserves_exact_file_text_and_parameter_types(project):
    path = project / "templates/engine/transformers/engine.py"
    spec = importlib.util.spec_from_file_location("transformer_template", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = module.parse_tool_calls(
        '<tool_call><function=replace_text><parameter=old>\n    return 1\n\n</parameter>'
        '<parameter=new>\n1\n</parameter></function></tool_call>')
    assert calls[0]["arguments"] == {"old": "    return 1\n", "new": "1"}


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
