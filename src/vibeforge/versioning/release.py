"""Freeze candidate sources and record the exact interpreter and evidence."""

import shutil
import os
from vibeforge.contracts.candidate import Candidate
from vibeforge.storage.atomic import digest, read_json, write_json
from vibeforge.versioning.artifacts import inventory, verify_release, source_ignore


def freeze(workspace, paths, parent, python, config_hash, evidence=()):
    files = inventory(workspace)
    source = digest(files)
    version = "v-" + digest({"source": source, "parent": parent, "python": python, "config": config_hash})[:20]
    destination = paths.release(version)
    manifest = Candidate(version, parent, source, files, python, config_hash, list(evidence)).to_dict()
    if destination.exists():
        verify_release(destination, read_json(destination / "manifest.json"))
        return read_json(destination / "manifest.json")
    staging = destination.with_name(destination.name + ".pending")
    if staging.exists():
        try:
            verify_release(staging, manifest)
            if read_json(staging / "manifest.json") != manifest:
                raise ValueError("Incomplete release manifest")
        except (ValueError, OSError):
            # Preserve interrupted materials, then build a fresh staging unit.
            import uuid
            staging.rename(staging.with_name(staging.name + ".interrupted-" + uuid.uuid4().hex[:8]))
    if not staging.exists():
        shutil.copytree(workspace, staging, ignore=source_ignore(workspace))
        write_json(staging / "manifest.json", manifest)
    verify_release(staging, manifest)
    for path in staging.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in [*sorted((p for p in staging.rglob("*") if p.is_dir()), reverse=True), staging]:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    staging.rename(destination)
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest
