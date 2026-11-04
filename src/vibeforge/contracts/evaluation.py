"""Evaluation outcomes consumed by the admission policy."""

from dataclasses import dataclass, asdict


@dataclass
class Decision:
    outcome: str
    reason: str
    gain: float | None = None
    lower: float | None = None
    upper: float | None = None

    def to_dict(self):
        return asdict(self)
