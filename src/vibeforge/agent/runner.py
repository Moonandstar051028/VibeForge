"""Bounded local-model tool loop and an explicitly labelled test-only runner."""

import json
import re
import time
import uuid
from pathlib import Path
from vibeforge.agent.tools import AgentTools
from vibeforge.engine.client import EngineClient
from vibeforge.observability.logs import append_record
from vibeforge.storage.atomic import read_json, write_json


def parse_object(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
    raise ValueError("Model did not return a JSON object")


def smoke_agent(client, max_new_tokens=80):
    response = client.chat(messages=[
        {"role": "system", "content": 'Return only valid JSON: {"tool":"submit_candidate","hypothesis":"ready"}'},
        {"role": "user", "content": "Confirm the agent response protocol. Do not explain."}],
        max_new_tokens=max_new_tokens)
    parsed = parse_object(response["text"])
    if parsed.get("tool") != "submit_candidate" or not parsed.get("hypothesis"):
        raise ValueError("Local model failed the agent JSON protocol smoke test")
    return {"passed": True, "response": parsed, "token_count": len(response.get("token_ids", []))}


def run_agent(client, context, workspace, trace, prompt_path, analysis_only=False):
    session = uuid.uuid4().hex
    settings = context["agent"]
    if settings.get("mode") == "scripted":
        actions = settings.get("actions", [])
        action = actions[min(context["iteration"] - 1, len(actions) - 1)] if actions else {}
        if not analysis_only:
            tools = AgentTools(workspace, trace)
            for edit in action.get("edits", []):
                tools.execute(edit["tool"], edit["arguments"])
        result = {"session_id": session, "mode": "scripted_test_fixture",
                  "hypothesis": action.get("hypothesis", "No code change; test fixture iteration."),
                  "previous_analysis": "The fixture reads the previous recorded decision; causal optimization ability is not tested.",
                  "summary": "Scripted integration fixture completed.", "complete": True}
        append_record(trace, result)
        return result
    tools = AgentTools(workspace, trace, context.get("evidence_root"))
    system = Path(prompt_path).read_text()
    if analysis_only:
        system += '\nOnly analyze the last experiment. Return {"tool":"submit_candidate","hypothesis":"summary","previous_analysis":"...","summary":"..."}. Do not edit files.'
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
    deadline = time.monotonic() + settings.get("timeout_seconds", 180)
    for turn in range(settings.get("max_turns", 6)):
        if time.monotonic() >= deadline:
            raise TimeoutError("Agent session budget exhausted")
        response = client.chat(messages=messages, max_new_tokens=settings.get("max_new_tokens", 512))
        append_record(trace, {"session_id": session, "turn": turn, "model_response": response})
        try:
            action = parse_object(response["text"])
            name = action.get("tool")
            if name == "submit_candidate":
                if not action.get("hypothesis"):
                    raise ValueError("Submission requires a hypothesis")
                if (context.get('previous') or {}).get('analysis_pending') and not action.get('previous_analysis'):
                    raise ValueError('Analyze the previous recorded result before submitting another candidate')
                return {**action, "session_id": session, "mode": "local", "complete": True}
            if analysis_only:
                raise ValueError("Only submission is allowed in analysis-only mode")
            result = tools.execute(name, action.get("arguments", {}))
            feedback = {"tool": name, "result": result}
        except (ValueError, KeyError, TypeError, OSError) as error:
            feedback = {"error": str(error), "instruction": "Return one valid JSON tool action."}
        messages.extend([{"role": "assistant", "content": response["text"]},
                         {"role": "user", "content": json.dumps(feedback, ensure_ascii=False)}])
    raise RuntimeError("Agent reached max_turns without submitting a candidate")


def worker(context_path, workspace, trace, result_path, url, prompt_path, analysis_only=False):
    context = read_json(context_path)
    result = run_agent(EngineClient(url, context["agent"].get("request_timeout", 180)),
                       context, Path(workspace), trace, prompt_path, analysis_only)
    write_json(result_path, result)
