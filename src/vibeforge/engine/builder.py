"""Compile candidate Python files in memory and validate their manifest."""

import ast
from pathlib import Path
from vibeforge.storage.atomic import read_json
from vibeforge.versioning.artifacts import inventory


def build(directory):
    directory = Path(directory)
    files = inventory(directory)
    if "engine.py" not in files or "engine.json" not in files:
        raise ValueError("An engine requires engine.py and engine.json")
    config = read_json(directory / "engine.json")
    for path in directory.rglob("*.py"):
        source = path.read_text()
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
    # Native build commands must be introduced as a separately bounded adapter,
    # rather than executing arbitrary engine.json command strings.
    return {"passed": True, "files": len(files), "backend": config.get("backend")}
