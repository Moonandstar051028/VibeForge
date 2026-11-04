"""Audited candidate-only file tools; absolute and escaping paths are rejected."""

import ast
from pathlib import Path
from vibeforge.observability.logs import append_record
from vibeforge.storage.atomic import atomic_write
from vibeforge.storage.paths import confined


class AgentTools:
    def __init__(self, workspace, trace, evidence_root=None):
        self.workspace, self.trace = Path(workspace).resolve(), trace
        self.evidence_root = Path(evidence_root).resolve() if evidence_root else None

    def execute(self, name, arguments):
        try:
            if name == "read_evidence":
                if self.evidence_root is None:
                    raise ValueError("No evidence root is configured")
                path = confined(self.evidence_root, arguments["path"])
                result = path.read_text()[:24000]
            elif name == "list_files":
                result = [str(p.relative_to(self.workspace)) for p in self.workspace.rglob("*") if p.is_file()]
            else:
                relative = arguments["path"]
                if Path(relative).is_absolute():
                    raise ValueError("Use candidate-relative paths")
                path = confined(self.workspace, relative)
                if name == "read_file":
                    result = path.read_text()[:24000]
                elif name == "write_file":
                    content = arguments["content"]
                    if len(content.encode()) > 512 * 1024:
                        raise ValueError("File exceeds tool size limit")
                    atomic_write(path, content)
                    result = {"written": relative}
                elif name == "replace_text":
                    old, new = arguments["old"], arguments["new"]
                    content = path.read_text()
                    if not old or content.count(old) != 1:
                        raise ValueError("Replacement must match exactly once")
                    atomic_write(path, content.replace(old, new, 1))
                    result = {"updated": relative}
                elif name == "check_python":
                    ast.parse(path.read_text(), filename=str(path))
                    result = {"syntax_valid": True}
                else:
                    raise ValueError(f"Unknown tool: {name}")
            append_record(self.trace, {"tool": name, "arguments": arguments, "result": result})
            return result
        except Exception as error:
            append_record(self.trace, {"tool": name, "arguments": arguments, "error": str(error)})
            raise
