"""Content-addressed source inventories and immutable release verification."""

import hashlib
from pathlib import Path
from vibeforge.storage.atomic import digest


EXCLUDED = {"__pycache__", ".git", ".venv", "build", ".cache"}


def source_ignore(root):
    root = Path(root).resolve()

    def ignored(directory, names):
        result = EXCLUDED.intersection(names)
        if Path(directory).resolve() == root and 'manifest.json' in names:
            result = result | {'manifest.json'}
        return result

    return ignored


def inventory(root):
    root = Path(root).resolve()
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED for part in relative.parts) or str(relative) == "manifest.json":
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not permitted in candidate sources: {relative}")
        if path.is_file():
            result[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def source_hash(root):
    return digest(inventory(root))


def verify_release(root, manifest):
    current = inventory(root)
    if current != manifest["files"] or digest(current) != manifest["source_hash"]:
        raise ValueError(f"Release was modified: {root}")
    return True
