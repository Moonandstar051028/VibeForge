"""Audited stdio MCP bridge for Codex evidence access and result submission."""

import argparse
import json
import os
import sys
from pathlib import Path

from jsonschema import ValidationError, validate

from vibeforge.agent.profile_tool import profile_inference, read_implementation
from vibeforge.agent.tools import AgentTools
from vibeforge.engine.client import EngineClient
from vibeforge.storage.atomic import write_json
from vibeforge.storage.paths import confined


TOOL_SCHEMAS = {
    "search_candidate": {
        "type": "object", "properties": {"path": {"type": "string"},
        "query": {"type": "string", "minLength": 1, "maxLength": 256},
        "offset": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
        "required": ["path", "query"], "additionalProperties": False,
    },
    "write_candidate": {
        "type": "object", "properties": {"path": {"type": "string"},
        "content": {"type": "string", "maxLength": 524288}},
        "required": ["path", "content"], "additionalProperties": False,
    },
    "read_candidate": {
        "type": "object", "properties": {"path": {"type": "string"},
        "pointer": {"type": "string", "minLength": 1, "maxLength": 512},
        "offset": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 6000}},
        "required": ["path"], "additionalProperties": False,
    },
    "replace_candidate": {
        "type": "object", "properties": {key: {"type": "string"} for key in ("path", "old", "new")},
        "required": ["path", "old", "new"], "additionalProperties": False,
    },
    "check_candidate": {
        "type": "object", "properties": {"path": {"type": "string"}},
        "required": ["path"], "additionalProperties": False,
    },
    "read_implementation": {
        "type": "object", "properties": {"symbol": {"type": "string"},
        "offset": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 12000}}, "additionalProperties": False,
    },
    "list_evidence": {
        "type": "object", "properties": {"path": {"type": "string"},
        "offset": {"type": "integer", "minimum": 0}}, "required": ["path"],
        "additionalProperties": False,
    },
    "read_evidence": {
        "type": "object", "properties": {"path": {"type": "string"},
        "pointer": {"type": "string", "minLength": 1, "maxLength": 512},
        "offset": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 12000}},
        "required": ["path"], "additionalProperties": False,
    },
    "search_history": {
        "type": "object", "properties": {"query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 10}},
        "required": ["query"], "additionalProperties": False,
    },
    "profile_inference": {
        "type": "object", "properties": {"prompt_index": {"type": "integer", "minimum": 0},
        "profile_id": {"type": "string"}}, "required": ["prompt_index"],
        "additionalProperties": False,
    },
    "read_telemetry": {
        "type": "object", "properties": {}, "additionalProperties": False,
    },
}

TOOL_DESCRIPTIONS = {
    "search_candidate": "Find literal text in a candidate source file; return bounded matching contexts and character offsets for read_candidate. The brief contains a Python definition index.",
    "write_candidate": "Create or replace a candidate-local source/config file (up to 512 KiB), including helper modules. Host context and evidence are read-only.",
    "read_candidate": "Read a candidate file or host context without a shell. Optional JSON pointer selects a field, e.g. /model. Offset/limit count characters within the selected text.",
    "replace_candidate": "Replace one exact unique text span in a candidate source/config file. Cannot modify host context or evidence.",
    "check_candidate": "Validate Python or JSON syntax by file extension, without executing candidate code. This does not establish runtime correctness.",
    "read_implementation": "List loaded model/helper symbols, or read one function by symbol with offset and limit. Read-only.",
    "list_evidence": "List bounded campaign-relative evidence paths.",
    "read_evidence": "Read a campaign-relative evidence file. Prefer optional JSON pointer for a known field, e.g. /mechanism or /correctness_precheck/checks, instead of paging through profiler dumps. Offset/limit count characters in the selected text.",
    "search_history": "Search recorded hypotheses, decisions, analyses and reflections.",
    "profile_inference": "Run one budgeted diagnostic profile against the stable engine and frozen workload.",
    "read_telemetry": "Read this Agent session's bounded ordinary-inference telemetry summary.",
}


def _reply(identifier, *, result=None, error=None):
    value = {"jsonrpc": "2.0", "id": identifier}
    if error is not None:
        value["error"] = error
    else:
        value["result"] = result
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve(schema_path, output_path, context_path=None, workspace=None, trace=None):
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    output_path = Path(output_path)
    context = json.loads(Path(context_path).read_text(encoding="utf-8")) if context_path else {}
    evidence = AgentTools(workspace or ".", trace or os.devnull, context.get("evidence_root"),
                          runtime_policy=context if context.get("optimization") is not None else None)
    profile_calls = []
    telemetry_calls = []
    available = ["record_result"]
    if workspace:
        available.extend(["read_candidate", "search_candidate"])
        if not context.get("analysis_only"):
            available.extend(["replace_candidate", "write_candidate", "check_candidate"])
    if context.get("evidence_root"):
        available.extend(["list_evidence", "read_evidence", "search_history"])
    if context.get("telemetry_session") and context.get("engine_url"):
        available.append("read_telemetry")
    profiling = (context.get("loop_protocol", {}).get("profiling") or {})
    if (context.get("candidate_contract") and context.get("engine_url")
            and profiling.get("enabled", True)):
        available.extend(["profile_inference", "read_implementation"])
    submitted = False
    exploration_calls = 0
    exploration_limit = min(8, max(1, int(context.get("agent", {}).get("max_turns", 20)) - 4))
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except ValueError:
            continue
        method = request.get("method")
        identifier = request.get("id")
        if method == "initialize":
            version = (request.get("params") or {}).get("protocolVersion", "2025-06-18")
            _reply(identifier, result={
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "vibeforge-submit", "version": "1.0.0"},
                "instructions": "Call record_result exactly once after finishing the assigned VibeForge work.",
            })
        elif method == "ping":
            _reply(identifier, result={})
        elif method == "tools/list":
            tools = [{
                "name": "record_result",
                "description": ("Record the one authoritative VibeForge result after candidate editing and "
                                "validation are complete. This tool validates the full result schema."),
                "inputSchema": schema,
            }]
            tools.extend({"name": name, "description": TOOL_DESCRIPTIONS[name],
                          "inputSchema": TOOL_SCHEMAS[name]}
                         for name in available if name != "record_result")
            _reply(identifier, result={"tools": tools})
        elif method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name")
            if name not in available:
                _reply(identifier, error={"code": -32601, "message": "Unknown tool"})
                continue
            if name != "record_result":
                arguments = params.get("arguments") or {}
                try:
                    validate(arguments, TOOL_SCHEMAS[name])
                    if name not in {"replace_candidate", "write_candidate", "check_candidate"}:
                        if exploration_calls >= exploration_limit:
                            raise ValueError(
                                "Exploration budget exhausted. Use the evidence already read; edit and check "
                                "one candidate, then record_result. In analysis-only mode, record the analysis now.")
                        exploration_calls += 1
                    if name in {"read_candidate", "search_candidate", "replace_candidate", "write_candidate", "check_candidate"}:
                        relative = confined(evidence.workspace, arguments["path"]).relative_to(evidence.workspace)
                        if name not in {"read_candidate", "search_candidate"} and relative.parts and relative.parts[0] in {
                            ".vibeforge", "campaigns", "final-analysis", "iterations", "knowledge-drafts",
                            "logs", "reports", "var",
                        }:
                            raise ValueError("Host context and evidence are read-only")
                        if name in {"replace_candidate", "write_candidate"} and (submitted or output_path.exists()):
                            raise ValueError("Cannot edit after recording a result")
                        result = evidence.execute({"read_candidate": "read_file", "search_candidate": "search_source", "replace_candidate": "replace_text",
                                                   "write_candidate": "write_file",
                                                   "check_candidate": "check_source"}[name], arguments)
                    elif name == "read_implementation":
                        result = read_implementation(context, trace or os.devnull, arguments)
                    elif name == "profile_inference":
                        result = profile_inference(context, trace or os.devnull, arguments, profile_calls)
                    elif name == "read_telemetry":
                        if len(telemetry_calls) >= 4:
                            raise ValueError("Telemetry read budget exhausted")
                        telemetry_calls.append(True)
                        report = EngineClient(
                            context["engine_url"], context.get("agent", {}).get("request_timeout", 180)
                        ).telemetry_report(context["telemetry_session"]["session_id"])
                        relative = context["telemetry_session"]["evidence_path"].replace(
                            ".json", f"-live-{len(telemetry_calls)}.json")
                        write_json(Path(context["evidence_root"]) / relative, report)
                        result = {"evidence_path": relative, "telemetry": report}
                    else:
                        result = evidence.execute(name, arguments)
                    rendered = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                    _reply(identifier, result={"content": [{"type": "text", "text": rendered}],
                                               "isError": False})
                except (OSError, ValidationError, ValueError, RuntimeError, SyntaxError) as error:
                    _reply(identifier, result={"content": [{"type": "text", "text":
                        f"VibeForge tool failed: {error}"}], "isError": True})
                continue
            if submitted or output_path.exists():
                _reply(identifier, result={"content": [{"type": "text", "text":
                    "A VibeForge result was already recorded."}], "isError": True})
                continue
            arguments = params.get("arguments")
            try:
                validate(arguments, schema)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = output_path.with_suffix(output_path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(arguments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                os.replace(temporary, output_path)
                submitted = True
            except (OSError, ValidationError, ValueError) as error:
                _reply(identifier, result={"content": [{"type": "text", "text":
                    f"VibeForge submission rejected: {error}"}], "isError": True})
                continue
            _reply(identifier, result={"content": [{"type": "text", "text":
                "VibeForge result recorded."}], "isError": False})
        elif identifier is not None:
            _reply(identifier, error={"code": -32601, "message": "Method not found"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context")
    parser.add_argument("--workspace")
    parser.add_argument("--trace")
    arguments = parser.parse_args()
    serve(arguments.schema, arguments.output, arguments.context, arguments.workspace, arguments.trace)


if __name__ == "__main__":
    main()
