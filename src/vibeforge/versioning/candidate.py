"""Create isolated candidate sources and a reviewable text diff."""

import difflib
import shutil
from pathlib import Path
from vibeforge.storage.atomic import atomic_write
from vibeforge.versioning.artifacts import inventory, source_ignore


def create_workspace(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Candidate workspace already exists: {destination}")
    inventory(source)
    shutil.copytree(source, destination, ignore=source_ignore(source))
    return destination


def write_diff(parent, candidate, output):
    parent, candidate = Path(parent), Path(candidate)
    changes = []
    for relative in sorted(set(inventory(parent)) | set(inventory(candidate))):
        left, right = parent / relative, candidate / relative
        old = left.read_text(errors="replace").splitlines(keepends=True) if left.exists() else []
        new = right.read_text(errors="replace").splitlines(keepends=True) if right.exists() else []
        changes.extend(difflib.unified_diff(old, new, fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    atomic_write(output, "".join(changes))
