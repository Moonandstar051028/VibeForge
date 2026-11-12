"""Responses API compatibility for the host-owned engine transport."""

import json
import time
import uuid


def _function_tools(request):
    """Flatten Codex namespaces for Chat Completions, retaining routing identity."""
    functions = {}
    for tool in request.get("tools") or []:
        namespace = tool.get("name") if tool.get("type") == "namespace" else None
        children = tool.get("tools", []) if namespace else [tool]
        for child in children:
            if child.get("type") != "function":
                continue
            name = f"{namespace}__{child['name']}" if namespace else child["name"]
            if name in functions:
                raise ValueError(f"Ambiguous flattened tool name: {name}")
            functions[name] = (namespace, child)
    return functions


def _chat_name(item):
    namespace = item.get("namespace")
    return f"{namespace}__{item['name']}" if namespace else item["name"]


def _content_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") in {"input_text", "output_text", "text"}
    )


def responses_to_chat(request):
    messages = []
    system_parts = []
    if request.get("instructions"):
        system_parts.append(request["instructions"])
    output_format = (request.get("text") or {}).get("format") or {}
    if output_format.get("type") in {"json_schema", "json_object"}:
        schema = output_format.get("schema") or output_format.get("json_schema", {}).get("schema")
        instruction = "Return only one JSON object"
        if schema:
            instruction += " matching this schema: " + json.dumps(schema, ensure_ascii=False)
        system_parts.append(instruction + ". Do not wrap it in Markdown.")
    items = request.get("input", [])
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            content = _content_text(item.get("content"))
            if role in {"developer", "system"}:
                system_parts.append(content)
            else:
                messages.append({"role": role, "content": content})
        elif kind == "function_call":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id") or item.get("id") or "call_" + uuid.uuid4().hex,
                    "type": "function",
                    "function": {"name": _chat_name(item), "arguments": item.get("arguments", "{}")},
                }],
            })
        elif kind == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"),
                             "content": _content_text(item.get("output", item.get("content", "")))})
    if system_parts:
        # Qwen chat templates require one system message at the beginning.
        # Responses exposes instructions, developer input and output formatting
        # separately, so combine those host-level constraints before inference.
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    tools = []
    for name, (_, tool) in _function_tools(request).items():
        tools.append({"type": "function", "function": {
            "name": name, "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        }})
    tool_choice = request.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        tool_choice = {"type": "function", "function": {"name": _chat_name(tool_choice)}}
    return {
        "messages": messages,
        "tools": tools or None,
        "tool_choice": tool_choice,
        "max_tokens": request.get("max_output_tokens"),
    }


def response_object(request, result):
    response_id = "resp_" + uuid.uuid4().hex
    output = []
    text = result.get("text") or ""
    if text:
        output.append({
            "id": "msg_" + uuid.uuid4().hex, "type": "message", "status": "completed",
            "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    functions = _function_tools(request)
    for call in result.get("tool_calls") or []:
        namespace, tool = functions.get(call["name"], (None, {"name": call["name"]}))
        output.append({
            "id": "fc_" + uuid.uuid4().hex, "type": "function_call", "status": "completed",
            "call_id": call.get("id") or "call_" + uuid.uuid4().hex,
            "name": tool["name"],
            **({"namespace": namespace} if namespace else {}),
            "arguments": call["arguments"] if isinstance(call["arguments"], str)
            else json.dumps(call["arguments"], ensure_ascii=False),
        })
    usage = result.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    completion_details = usage.get("completion_tokens_details") or {}
    finish_reason = result.get("finish_reason", "stop")
    incomplete_reason = "max_output_tokens" if finish_reason == "length" else None
    failed_reasoning = finish_reason == "incomplete_thinking"
    status = "failed" if failed_reasoning else (
        "incomplete" if incomplete_reason else "completed")
    error = ({
        "code": "server_error",
        "message": "The local model stopped before closing its thinking segment.",
        "param": None,
        "type": "incomplete_thinking",
    } if failed_reasoning else None)
    return {
        "id": response_id, "object": "response", "created_at": int(time.time()),
        "status": status, "completed_at": int(time.time()) if status == "completed" else None,
        "model": request.get("model", "vibeforge-local"), "output": output,
        "output_text": text, "error": error,
        "incomplete_details": ({"reason": incomplete_reason} if status == "incomplete" else None),
        "parallel_tool_calls": bool(request.get("parallel_tool_calls", True)),
        "tool_choice": request.get("tool_choice", "auto"), "tools": request.get("tools") or [],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "output_tokens_details": {
                      "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
                  },
                  "total_tokens": input_tokens + output_tokens},
    }


def response_events(response):
    sequence = 0

    def event(kind, **fields):
        nonlocal sequence
        value = {"type": kind, "sequence_number": sequence, **fields}
        sequence += 1
        return value

    pending = dict(response, status="in_progress", output=[])
    yield event("response.created", response=pending)
    yield event("response.in_progress", response=pending)
    for index, item in enumerate(response["output"]):
        if item["type"] == "message":
            added = dict(item, status="in_progress", content=[])
            yield event("response.output_item.added", output_index=index, item=added)
            part = item["content"][0]
            yield event("response.content_part.added", item_id=item["id"], output_index=index,
                        content_index=0, part={"type": "output_text", "text": "", "annotations": []})
            if part["text"]:
                yield event("response.output_text.delta", item_id=item["id"], output_index=index,
                            content_index=0, delta=part["text"])
            yield event("response.output_text.done", item_id=item["id"], output_index=index,
                        content_index=0, text=part["text"])
            yield event("response.content_part.done", item_id=item["id"], output_index=index,
                        content_index=0, part=part)
        else:
            yield event("response.output_item.added", output_index=index,
                        item={**item, "status": "in_progress", "arguments": ""})
            if item["arguments"]:
                yield event("response.function_call_arguments.delta", item_id=item["id"],
                            output_index=index, delta=item["arguments"])
            yield event("response.function_call_arguments.done", item_id=item["id"],
                        output_index=index, name=item["name"], arguments=item["arguments"])
        yield event("response.output_item.done", output_index=index, item=item)
    terminal_event = {
        "completed": "response.completed",
        "failed": "response.failed",
    }.get(response["status"], "response.incomplete")
    yield event(terminal_event, response=response)
