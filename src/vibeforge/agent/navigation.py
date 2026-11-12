"""Bounded source navigation without executing candidate code."""

import ast
from pathlib import Path

from vibeforge.storage.paths import confined
from vibeforge.versioning.artifacts import EXCLUDED


def candidate_index(workspace):
    """Expose actual file names and Python definition offsets, not suggestions."""
    root = Path(workspace).resolve()
    files = []
    definitions = []
    errors = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED or part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        confined(root, str(relative))
        if len(files) >= 64:
            break
        files.append(str(relative))
        if path.suffix != ".py" or path.stat().st_size > 512 * 1024:
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (ValueError, SyntaxError, OSError) as error:
            errors.append({"path": str(relative), "error": str(error)[:200]})
            continue
        offsets = [0]
        for line in source.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))

        def collect(body, prefix=""):
            for node in body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if len(definitions) >= 80:
                    return
                name = prefix + node.name
                start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
                definitions.append({
                    "path": str(relative), "symbol": name, "line": start,
                    "offset": offsets[start - 1], "end_offset": offsets[node.end_lineno],
                })
                if isinstance(node, ast.ClassDef):
                    collect(node.body, name + ".")

        collect(tree.body)
    return {"files": files, "definitions": definitions, "syntax_errors": errors,
            "limits": {"files": 64, "definitions": 80},
            "usage": "Offsets count characters and can be passed directly to read_candidate. "
                     "Use search_candidate for a literal symbol or configuration field."}


def search_source(path, query, *, offset=0, limit=12):
    """Find literal text and return bounded contexts with exact read offsets."""
    if not isinstance(query, str) or not query or len(query) > 256:
        raise ValueError("query must contain 1..256 characters")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative character offset")
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("limit must be 1..20")
    if path.stat().st_size > 512 * 1024:
        raise ValueError("Source search file exceeds 512 KiB")
    source = path.read_text(encoding="utf-8")
    matches = []
    cursor = offset
    while len(matches) < limit:
        found = source.find(query, cursor)
        if found < 0:
            break
        start = max(0, found - 160)
        end = min(len(source), found + len(query) + 200)
        matches.append({"offset": found, "line": source.count("\n", 0, found) + 1,
                        "context_offset": start, "context": source[start:end]})
        cursor = found + len(query)
    next_offset = cursor if source.find(query, cursor) >= 0 else None
    return {"matches": matches, "next_offset": next_offset, "total_characters": len(source),
            "search": "literal, case-sensitive; offsets count characters"}
