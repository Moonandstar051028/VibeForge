"""Run the Qwen-Agent Assistant against the campaign's local inference server."""

import json
import tempfile
import time
import uuid
from contextlib import closing
from pathlib import Path

from vibeforge.agent.tools import AgentTools
from vibeforge.observability.logs import append_record
from vibeforge.storage.atomic import read_json, write_json


TOOL_SCHEMAS = {
    "read_file": ("Read a candidate-relative source file.", {"path": "string"}, ["path"]),
    "replace_text": ("Replace one exact, unique text span in a candidate file.",
                     {"path": "string", "old": "string", "new": "string"}, ["path", "old", "new"]),
    "write_file": ("Write a file inside the candidate workspace.",
                   {"path": "string", "content": "string"}, ["path", "content"]),
    "list_files": ("List files in the candidate workspace.", {}, []),
    "check_python": ("Parse a Python file to check its syntax.", {"path": "string"}, ["path"]),
    "read_evidence": ("Read a campaign evidence file by evidence-root-relative path.",
                      {"path": "string"}, ["path"]),
    "submit_candidate": ("Submit the hypothesis, previous-result analysis, and change summary.", {
        "hypothesis": "string", "previous_analysis": "string", "summary": "string"
    }, ["hypothesis"]),
}


def _make_tool(name, callback):
    from qwen_agent.tools.base import BaseTool

    description, properties, required = TOOL_SCHEMAS[name]
    parameters = {
        "type": "object",
        "properties": {key: {"type": value} for key, value in properties.items()},
        "required": required,
    }

    def call(self, params, **kwargs):
        arguments = self._verify_json_format_args(params, strict_json=True)
        return json.dumps(callback(arguments), ensure_ascii=False)

    tool_type = type(
        f"VibeForge_{name}",
        (BaseTool,),
        {"name": name, "description": description, "parameters": parameters, "call": call},
    )
    return tool_type()


def _assistant_tools(workspace, trace, context, submission, analysis_only=False, smoke=False):
    tools = AgentTools(workspace, trace, context.get("evidence_root"))
    names = ["submit_candidate"] if smoke else ["read_file", "list_files", "submit_candidate"]
    if not analysis_only and not smoke:
        names.extend(["replace_text", "write_file", "check_python"])
    if not smoke and context.get("evidence_root"):
        names.append("read_evidence")

    def execute(name):
        def callback(arguments):
            if name != "submit_candidate":
                return tools.execute(name, arguments)
            try:
                hypothesis = arguments["hypothesis"].strip()
                if not hypothesis:
                    raise ValueError("Submission requires a hypothesis")
                previous_analysis = arguments.get("previous_analysis", "").strip()
                if (context.get("previous") or {}).get("analysis_pending") and not previous_analysis:
                    raise ValueError("Analyze the previous recorded result before submitting another candidate")
                submission.update({
                    "hypothesis": hypothesis,
                    "previous_analysis": previous_analysis,
                    "summary": arguments.get("summary", "").strip(),
                    "complete": True,
                })
                append_record(trace, {"tool": name, "arguments": arguments, "result": {"submitted": True}})
                return {"submitted": True}
            except Exception as error:
                append_record(trace, {"tool": name, "arguments": arguments, "error": str(error)})
                raise
        return callback

    return [_make_tool(name, execute(name)) for name in names]


def _run_qwen_agent(url, context, workspace, trace, prompt, api_key, analysis_only=False, smoke=False):
    from qwen_agent.agents import Assistant

    settings = context.get("agent", {})
    session_id = uuid.uuid4().hex
    submission = {}
    system = prompt
    if analysis_only:
        system += "\nAnalyze the recorded experiment only. Do not edit files. Call submit_candidate with the causal analysis."
    if smoke:
        system = "Call submit_candidate exactly once with a short readiness hypothesis. Do not use other tools."
    model = context.get("model", {}).get("name") or "vibeforge-local"
    assistant = Assistant(
        llm={
            "model": model,
            "model_type": "oai",
            "model_server": url.rstrip("/") + "/v1",
            "api_key": api_key or "EMPTY",
            "generate_cfg": {
                "use_raw_api": True,
                "max_tokens": settings.get("max_new_tokens", 512),
                "request_timeout": settings.get("request_timeout", 180),
                "temperature": 0,
            },
        },
        function_list=_assistant_tools(workspace, trace, context, submission, analysis_only, smoke),
        system_message=system,
    )
    messages = [{"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
    deadline = time.monotonic() + settings.get("timeout_seconds", 180)
    with closing(assistant.run(messages=messages)) as updates:
        for state in updates:
            snapshot = [item.model_dump() if hasattr(item, "model_dump") else item for item in state]
            append_record(trace, {"session_id": session_id, "event": "qwen_agent_update", "messages": snapshot})
            if time.monotonic() >= deadline:
                raise TimeoutError("Qwen-Agent session budget exhausted")
            if submission.get("complete"):
                break
            completed_calls = sum(item.get("role") == "function" for item in snapshot)
            if completed_calls >= settings.get("max_turns", 6):
                raise RuntimeError("Qwen-Agent reached the tool-call budget without a valid submission")
    if not submission.get("complete"):
        raise RuntimeError("Qwen-Agent finished without calling submit_candidate")
    return {**submission, "session_id": session_id, "mode": "qwen_agent", "complete": True}


def smoke_agent(client, max_new_tokens=80, trace=None):
    with tempfile.TemporaryDirectory(prefix="vibeforge-qwen-agent-smoke-") as directory:
        root = Path(directory)
        result = _run_qwen_agent(
            client.url,
            {"agent": {"max_new_tokens": max_new_tokens, "request_timeout": client.timeout}},
            root,
            trace or root / "trace.jsonl",
            "",
            "vibeforge-smoke",
            smoke=True,
        )
    return {"passed": True, "framework": "Qwen-Agent", "submission": result}


def run_agent(url, context, workspace, trace, prompt_path, api_key="", analysis_only=False):
    if context.get("agent", {}).get("mode") == "scripted":
        action_list = context["agent"].get("actions", [])
        action = action_list[min(context["iteration"] - 1, len(action_list) - 1)] if action_list else {}
        if not analysis_only:
            tools = AgentTools(workspace, trace)
            for edit in action.get("edits", []):
                tools.execute(edit["tool"], edit["arguments"])
        result = {
            "session_id": uuid.uuid4().hex,
            "mode": "scripted_test_fixture",
            "hypothesis": action.get("hypothesis", "No code change; test fixture iteration."),
            "previous_analysis": "The fixture records a predetermined decision; causal optimization ability is not tested.",
            "summary": "Scripted integration fixture completed.",
            "complete": True,
        }
        append_record(trace, result)
        return result

    context = dict(context)
    context["agent"] = dict(context.get("agent", {}))
    framework = context["agent"].get("framework", "qwen-agent")
    if framework != "qwen-agent":
        raise ValueError(f"Unsupported Agent framework: {framework}")
    context["agent"]["mode"] = "local"
    system = Path(prompt_path).read_text()
    return _run_qwen_agent(url, context, Path(workspace), trace, system, api_key, analysis_only)


def worker(context_path, workspace, trace, result_path, url, prompt_path, api_key="", analysis_only=False, smoke=False):
    context = read_json(context_path)
    if smoke:
        with tempfile.TemporaryDirectory(prefix="vibeforge-agent-smoke-") as directory:
            result = _run_qwen_agent(url, context, Path(directory), trace, "", api_key, smoke=True)
        result = {"passed": True, "framework": "Qwen-Agent", "submission": result}
    else:
        result = run_agent(url, context, Path(workspace), trace, prompt_path, api_key, analysis_only)
    write_json(result_path, result)
