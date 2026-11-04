"""Immutable release manifest contract."""

from dataclasses import dataclass, field, asdict


@dataclass
class Candidate:
    version_id: str
    parent: str | None
    source_hash: str
    files: dict
    runtime_python: str
    config_hash: str
    evidence: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
