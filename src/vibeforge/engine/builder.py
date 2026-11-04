"""Compile candidate Python files in memory and validate their manifest."""

import ast
import json
import os
import subprocess
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
    return {"passed": True, "files": len(files), "backend": config.get("backend")}


def build_worker(directory, cache, command, device="cpu"):
    """Run the campaign's frozen build recipe with writes confined to scratch/source."""
    from vibeforge.engine.sandbox import restrict_writes, prepare_device
    directory, cache = Path(directory).resolve(), Path(cache).resolve()
    build(directory)
    prepare_device(device)
    restrict_writes([directory, cache])
    argv = json.loads(command)
    if argv:
        subprocess.run(argv, cwd=directory, check=True, env=dict(os.environ))
    return build(directory)
