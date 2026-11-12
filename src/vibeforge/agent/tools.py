"""Audited candidate-only file tools; absolute and escaping paths are rejected."""

import ast
import json
from pathlib import Path
from vibeforge.observability.logs import append_record
from vibeforge.storage.atomic import atomic_write
from vibeforge.storage.paths import confined


class AgentTools:
    def __init__(self, workspace, trace, evidence_root=None, runtime_policy=None):
        self.workspace, self.trace = Path(workspace).resolve(), trace
        self.evidence_root = Path(evidence_root).resolve() if evidence_root else None
        self.runtime_policy = runtime_policy
        config = self.workspace / "engine.json"
        self.engine_baseline = json.loads(config.read_text()) if runtime_policy is not None and config.is_file() else None

    def _write(self, path, content):
        if self.engine_baseline is not None and path == self.workspace / "engine.json":
            from vibeforge.contracts.precision import validate_candidate_runtime
            try:
                candidate = json.loads(content)
                if not isinstance(candidate, dict):
                    raise ValueError("engine.json must be a JSON object")
                validate_candidate_runtime(self.engine_baseline, candidate, self.runtime_policy)
            except ValueError as error:
                raise ValueError(
                    f"Engine config write rejected; the file is unchanged: {error}. "
                    "Preserve all fixed fields. Use an exact replacement of the intended field, "
                    "or include the complete existing configuration.") from error
        atomic_write(path, content)

    def execute(self, name, arguments):
        try:
            if name == "search_history":
                from vibeforge.agent.memory import search_memory
                if self.evidence_root is None:
                    raise ValueError("No evidence root is configured")
                result = search_memory(self.evidence_root, arguments.get("query", ""), arguments.get("limit", 5))
            elif name == "read_evidence":
                if self.evidence_root is None:
                    raise ValueError("No evidence root is configured")
                path = confined(self.evidence_root, arguments["path"])
                result = self._read_limited(path, arguments.get("limit", 3000), arguments.get("offset", 0),
                                            pointer=arguments.get("pointer"), ceiling=12000)
            elif name == "list_evidence":
                if self.evidence_root is None:
                    raise ValueError("No evidence root is configured")
                directory = (self.evidence_root.resolve() if arguments["path"] in ("", ".")
                             else confined(self.evidence_root, arguments["path"]))
                if not directory.is_dir():
                    if directory.is_file():
                        raise ValueError("This is an evidence file. Use read_evidence with the same path; "
                                         "list_evidence accepts directories only.")
                    raise ValueError("Evidence path must be an existing directory")
                files = sorted(
                    str(path.relative_to(self.evidence_root))
                    for path in directory.rglob("*")
                    if path.is_file() and not path.is_symlink()
                )
                offset = max(0, int(arguments.get("offset", 0)))
                result = {"files": files[offset:offset + 100], "total": len(files),
                          "next_offset": offset + 100 if offset + 100 < len(files) else None}
            elif name == "list_files":
                result = [str(p.relative_to(self.workspace)) for p in self.workspace.rglob("*") if p.is_file()]
            else:
                relative = arguments["path"]
                if Path(relative).is_absolute():
                    raise ValueError("Use candidate-relative paths")
                path = confined(self.workspace, relative)
                if name in {"read_file", "search_source", "check_python", "check_source", "replace_text"} and not path.exists():
                    if self.evidence_root is not None:
                        try:
                            evidence_path = confined(self.evidence_root, relative)
                        except ValueError:
                            evidence_path = None
                        if evidence_path is not None and evidence_path.is_file():
                            raise ValueError("This path belongs to campaign evidence, not the candidate workspace. "
                                             "Use read_evidence with the same path; evidence is read-only.")
                    matches = [str(p.relative_to(self.workspace))
                               for p in sorted(self.workspace.rglob(Path(relative).name))
                               if p.is_file() and not p.is_symlink()][:8]
                    raise ValueError(f"Candidate file does not exist: {relative}. "
                                     f"Matching workspace-relative file names: {matches}. "
                                     "Consult candidate_files and candidate_source_index in the brief.")
                if name == "read_file":
                    result = self._read_limited(path, arguments.get("limit", 3000), arguments.get("offset", 0),
                                                pointer=arguments.get("pointer"))
                elif name == "search_source":
                    from vibeforge.agent.navigation import search_source
                    result = search_source(path, arguments["query"],
                                           offset=arguments.get("offset", 0),
                                           limit=arguments.get("limit", 12))
                elif name == "write_file":
                    content = arguments["content"]
                    if len(content.encode()) > 512 * 1024:
                        raise ValueError("File exceeds tool size limit")
                    self._write(path, content)
                    result = {"written": relative}
                elif name == "replace_text":
                    old, new = arguments["old"], arguments["new"]
                    content = path.read_text()
                    if not old or content.count(old) != 1:
                        raise ValueError("Replacement must match exactly once")
                    self._write(path, content.replace(old, new, 1))
                    result = {"updated": relative}
                elif name == "check_python":
                    ast.parse(path.read_text(), filename=str(path))
                    result = {"syntax_valid": True}
                elif name == "check_source":
                    if path.suffix == ".py":
                        compile(path.read_text(), str(path), "exec")
                        language = "python"
                    elif path.suffix == ".json":
                        json.loads(path.read_text())
                        language = "json"
                    else:
                        raise ValueError("Syntax checks support .py and .json candidate files")
                    result = {"syntax_valid": True, "language": language,
                              "scope": "syntax only; runtime and performance are not validated"}
                else:
                    raise ValueError(f"Unknown tool: {name}")
            append_record(self.trace, {"tool": name, "arguments": arguments, "result": result})
            return result
        except Exception as error:
            append_record(self.trace, {"tool": name, "arguments": arguments, "error": str(error)})
            raise

    @staticmethod
    def _read_limited(path, limit=3000, offset=0, *, pointer=None, ceiling=6000):
        limit, offset = max(1, min(int(limit), ceiling)), max(0, int(offset))
        content = path.read_text()
        if pointer is not None:
            if not isinstance(pointer, str) or not pointer.startswith("/") or len(pointer) > 512:
                raise ValueError("pointer must be a JSON Pointer starting with '/', at most 512 characters")
            value = json.loads(content)
            for raw in pointer[1:].split("/"):
                import re
                if re.search(r"~(?![01])", raw):
                    raise ValueError("JSON Pointer escapes must be ~0 or ~1")
                key = raw.replace("~1", "/").replace("~0", "~")
                if isinstance(value, dict) and key in value:
                    value = value[key]
                elif isinstance(value, list) and (key == "0" or (key.isascii() and key.isdigit() and not key.startswith("0"))) and int(key) < len(value):
                    value = value[int(key)]
                else:
                    available = list(value)[:20] if isinstance(value, dict) else f"array length {len(value)}" if isinstance(value, list) else "scalar value"
                    raise ValueError(f"JSON Pointer component {key!r} not found; available: {available}")
            content = json.dumps(value, ensure_ascii=False, indent=2)
        if len(content) > offset + limit:
            return content[offset:offset + limit] + (
                f"\n[Characters {offset}:{offset + limit}/{len(content)}. "
                f"Next page: offset={offset + limit}; continue there, not at 0.]")
        return content[offset:offset + limit]
