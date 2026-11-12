import importlib.util
import json

import pytest

from vibeforge.agent.tools import AgentTools
from vibeforge.agent.profile_tool import profile_inference
from vibeforge.knowledge.validator import validate_rule
from vibeforge.knowledge.registry import materialize_evidence
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


def test_evidence_listing_is_scoped_to_campaign_root(tmp_path):
    evidence = tmp_path / "campaign"
    nested = evidence / "iterations" / "0001"
    nested.mkdir(parents=True)
    (nested / "result.json").write_text("{}")
    (tmp_path / "outside.json").write_text("{}")
    tools = AgentTools(tmp_path / "workspace", tmp_path / "commands.jsonl", evidence)

    assert tools.execute("list_evidence", {"path": "iterations/0001"}) == {
        "files": ["iterations/0001/result.json"], "total": 1, "next_offset": None}
    with pytest.raises(ValueError, match="escapes"):
        tools.execute("read_evidence", {"path": "../outside.json"})


def test_evidence_listing_requires_campaign_root(tmp_path):
    tools = AgentTools(tmp_path / "workspace", tmp_path / "commands.jsonl")
    with pytest.raises(ValueError, match="No evidence root"):
        tools.execute("list_evidence", {"path": "iterations/0001"})


def test_large_evidence_reads_are_bounded(tmp_path):
    evidence = tmp_path / "campaign"
    evidence.mkdir()
    (evidence / "large.json").write_text("x" * 7000)
    tools = AgentTools(tmp_path / "workspace", tmp_path / "commands.jsonl", evidence)

    result = tools.execute("read_evidence", {"path": "large.json"})
    assert len(result) < 3100
    assert "Next page: offset=3000" in result


def test_global_knowledge_evidence_is_materialized_inside_campaign(tmp_path):
    project = tmp_path / "project"
    campaign = project / "var/campaigns/current"
    source = project / "var/campaigns/old/iterations/0001/probe.json"
    source.parent.mkdir(parents=True)
    campaign.mkdir(parents=True)
    source.write_text('{"measured": true}')
    rules = [{"rule_id": "old-failure", "evidence": [{
        "path": "var/campaigns/old/iterations/0001/probe.json"}]}]

    result = materialize_evidence(rules, project, campaign)

    copied = result[0]["evidence"][0]
    assert copied["source_path"] == "var/campaigns/old/iterations/0001/probe.json"
    assert copied["path"].startswith("knowledge/evidence/")
    assert (campaign / copied["path"]).read_text() == '{"measured": true}'
    assert "source_path" not in rules[0]["evidence"][0]


def test_shared_profile_tool_binds_frozen_workload_and_enforces_session_budget(tmp_path, monkeypatch):
    observed = []

    class Client:
        def __init__(self, url, timeout):
            observed.append({"url": url, "timeout": timeout})

        def profile(self, **arguments):
            observed[-1]["arguments"] = arguments
            return {"kind": "host_inference_profile", "measurements": {}}

    monkeypatch.setattr("vibeforge.agent.profile_tool.EngineClient", Client)
    context = {
        "engine_url": "http://127.0.0.1:9000", "stable_ref": "v-stable",
        "profile_hash": "profile", "iteration": 4, "evidence_root": str(tmp_path),
        "agent": {"request_timeout": 12},
        "loop_protocol": {"profiling": {"max_new_tokens": 64, "timeout_seconds": 3,
                                           "max_profiles_per_session": 1}},
        "workload": {"observation_profile": "long", "profiles": [{
            "id": "long", "prompts": ["profile this prompt"], "max_new_tokens": 96,
            "concurrency": 1, "warmup": 0, "repeats": 1,
        }]},
        "candidate_contract": {"observation_paths": []},
    }
    calls = []

    result = profile_inference(context, tmp_path / "commands.jsonl",
                               {"profile_id": "long", "prompt_index": 0}, calls)

    assert observed[0]["arguments"] == {"prompt": "profile this prompt", "max_new_tokens": 64}
    assert observed[0]["timeout"] == 3
    assert result["evidence_path"] == "observations/v-stable/agent-4-1.json"
    assert context["candidate_contract"]["observation_paths"] == [result["evidence_path"]]
    assert (tmp_path / result["evidence_path"]).is_file()
    with pytest.raises(ValueError, match="budget exhausted"):
        profile_inference(context, tmp_path / "commands.jsonl",
                          {"profile_id": "long", "prompt_index": 0}, calls)


def test_shared_profile_tool_honors_disabled_configuration(tmp_path):
    context = {
        "loop_protocol": {"profiling": {"enabled": False}},
        "workload": {"profiles": [{
            "id": "default", "prompts": ["prompt"], "max_new_tokens": 8,
            "concurrency": 1, "warmup": 0, "repeats": 1,
        }]},
    }

    with pytest.raises(ValueError, match="disabled"):
        profile_inference(context, tmp_path / "commands.jsonl",
                          {"profile_id": "default", "prompt_index": 0}, [])


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
    assert result["usage"] == {
        "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
        "completion_tokens_details": {"reasoning_tokens": 0, "visible_tokens": 1},
    }


def test_transformers_chat_completion_enables_thinking_and_reports_tokens(project, monkeypatch):
    path = project / "templates/engine/transformers/engine.py"
    spec = importlib.util.spec_from_file_location("transformer_thinking", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            observed["thinking"] = kwargs["enable_thinking"]
            return "rendered prompt"

        def encode(self, text, **kwargs):
            assert text == "</think>"
            return [99]

    def generate(prompt, **kwargs):
        observed["generation_options"] = kwargs["generation_options"]
        return {"text": "inspect evidence</think>done<|im_end|>",
                "input_ids": [1, 2], "token_ids": [10, 11, 99, 12, 13],
                "finish_reason": "stop"}

    monkeypatch.setattr(module, "TOKENIZER", Tokenizer())
    monkeypatch.setattr(module, "CONFIG", {"thinking": {"enabled": True, "temperature": 0.6,
                                                        "top_p": 0.95, "top_k": 20, "min_p": 0.0}})
    monkeypatch.setattr(module, "generate", generate)

    result = module.chat_completion([{"role": "user", "content": "work"}], max_tokens=64)

    assert observed == {"thinking": True, "generation_options": {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}}
    assert result["reasoning_content"] == "inspect evidence"
    assert result["text"] == "done"
    assert result["thinking"] == {"enabled": True, "complete": True,
                                  "recovered_without_thinking": False,
                                  "reasoning_tokens": 3, "visible_tokens": 2}


def test_transformers_chat_completion_retries_unclosed_thinking_without_prefix(project, monkeypatch):
    path = project / "templates/engine/transformers/engine.py"
    spec = importlib.util.spec_from_file_location("transformer_unclosed_thinking", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "rendered prompt"

        def encode(self, text, **kwargs):
            return [99]

    monkeypatch.setattr(module, "TOKENIZER", Tokenizer())
    monkeypatch.setattr(module, "CONFIG", {"thinking": {"enabled": True, "temperature": 0.6}})
    monkeypatch.setattr(module, "generate", lambda prompt, **kwargs: {
        "text": "still reasoning", "input_ids": [1, 2], "token_ids": [10, 11],
        "finish_reason": "stop",
    })

    result = module.chat_completion([{"role": "user", "content": "work"}], max_tokens=64)

    assert result["text"] == "still reasoning"
    assert result["thinking"]["complete"] is True
    assert result["thinking"]["recovered_without_thinking"] is True
    assert result["finish_reason"] == "stop"


def test_transformers_chat_completion_recovers_from_unclosed_thinking(project, monkeypatch):
    path = project / "templates/engine/transformers/engine.py"
    spec = importlib.util.spec_from_file_location("transformer_thinking_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            observed.append(kwargs["enable_thinking"])
            return "thinking prompt" if kwargs["enable_thinking"] else "plain prompt"

        def encode(self, text, **kwargs):
            return [99]

    def generate(prompt, **kwargs):
        if prompt == "thinking prompt":
            return {"text": "<|im_end|>", "input_ids": [1], "token_ids": [2],
                    "finish_reason": "stop"}
        return {"text": "<tool_call><function=record_result><parameter=ready>true</parameter></function></tool_call>",
                "input_ids": [1], "token_ids": [3, 4], "finish_reason": "stop"}

    monkeypatch.setattr(module, "TOKENIZER", Tokenizer())
    monkeypatch.setattr(module, "CONFIG", {"thinking": {"enabled": True, "temperature": 0.6}})
    monkeypatch.setattr(module, "generate", generate)
    tools = [{"type": "function", "function": {"name": "record_result", "parameters": {
        "type": "object", "properties": {"ready": {"type": "boolean"}}}}}]

    result = module.chat_completion(
        [{"role": "user", "content": "submit"}], tools=tools, max_tokens=64,
    )

    assert observed == [True, False]
    assert result["finish_reason"] == "stop"
    assert result["tool_calls"][0]["name"] == "record_result"
    assert result["thinking"]["recovered_without_thinking"] is True


def test_transformers_quality_chat_stays_non_thinking(project, monkeypatch):
    path = project / "templates/engine/transformers/engine.py"
    spec = importlib.util.spec_from_file_location("transformer_quality_chat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            observed.update(kwargs)
            return "prompt"

    monkeypatch.setattr(module, "TOKENIZER", Tokenizer())
    monkeypatch.setattr(module, "CONFIG", {"thinking": {"enabled": True}})
    monkeypatch.setattr(module, "generate", lambda prompt, max_new_tokens: {"text": "answer"})

    assert module.chat([{"role": "user", "content": "question"}], 16)["text"] == "answer"
    assert observed["enable_thinking"] is False


def test_transformers_chat_completion_limits_required_tool_choice(project, monkeypatch):
    path = project / "templates/engine/transformers/engine.py"
    spec = importlib.util.spec_from_file_location("transformer_required_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools = [
        {"type": "function", "function": {"name": name, "parameters": {"properties": {"hypothesis": {"type": "string"}}}}}
        for name in ("read_file", "submit_candidate")
    ]

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["submit_candidate"]
            assert "must call the submit_candidate function" in messages[0]["content"]
            return "rendered prompt"

    monkeypatch.setattr(module, "TOKENIZER", Tokenizer())
    monkeypatch.setattr(module, "generate", lambda prompt, **kwargs: {
        "text": "<tool_call><function=submit_candidate><parameter=hypothesis>test</parameter></function></tool_call>",
        "input_ids": [1], "token_ids": [2],
    })
    result = module.chat_completion(
        [{"role": "system", "content": "recover"}, {"role": "user", "content": "submit"}],
        tools=tools,
        max_tokens=32,
        tool_choice={"type": "function", "function": {"name": "submit_candidate"}},
    )
    assert result["tool_calls"][0]["name"] == "submit_candidate"


def test_native_tool_parser_preserves_exact_file_text_and_parameter_types(project):
    path = project / "templates/engine/transformers/engine.py"
    spec = importlib.util.spec_from_file_location("transformer_template", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = module.parse_tool_calls(
        '<tool_call><function=replace_text><parameter=old>\n    return 1\n\n</parameter>'
        '<parameter=new>\n1\n</parameter></function></tool_call>')
    assert calls[0]["arguments"] == {"old": "    return 1\n", "new": "1"}


def test_native_tool_parser_accepts_python_literals_from_local_models(project):
    path = project / "templates/engine/transformers/engine.py"
    spec = importlib.util.spec_from_file_location("transformer_local_literals", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools = [{"type": "function", "function": {"name": "record_result", "parameters": {
        "type": "object", "properties": {
            "blocked": {"type": "boolean"}, "reflection": {"type": "object"},
        }}}}]
    text = ("<tool_call><function=record_result>"
            "<parameter=blocked>False</parameter>"
            "<parameter=reflection>{'iteration': 1, 'status': 'inconclusive'}</parameter>"
            "</function></tool_call>")

    calls = module.parse_tool_calls(text, tools)

    assert calls == [{"name": "record_result", "arguments": {
        "blocked": False, "reflection": {"iteration": 1, "status": "inconclusive"}}}]


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
    rule["status"] = "invalidated"
    assert retrieve([rule], {"device": "thor"}) == [rule]
