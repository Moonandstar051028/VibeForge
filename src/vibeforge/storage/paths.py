"""Project and configurable persistent-data paths."""

from dataclasses import dataclass
from pathlib import Path
import re


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def confined(root, relative):
    root = Path(root).resolve()
    destination = (root / relative).resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError(f"Path escapes workspace: {relative}")
    return destination


@dataclass(frozen=True)
class Paths:
    project: Path
    data: Path

    @classmethod
    def create(cls, project, data=None):
        project = Path(project).resolve()
        data = Path(data or project / "var")
        if not data.is_absolute():
            data = project / data
        result = cls(project, data.resolve())
        for name in ("state", "profiles", "workspaces", "releases", "campaigns", "cache"):
            (result.data / name).mkdir(parents=True, exist_ok=True)
        return result

    def campaign(self, campaign_id):
        return self.data / "campaigns" / identifier(campaign_id)

    def workspace(self, campaign_id, iteration):
        return self.data / "workspaces" / identifier(campaign_id) / f"{iteration:04d}"

    def release(self, version):
        return self.data / "releases" / identifier(version)
