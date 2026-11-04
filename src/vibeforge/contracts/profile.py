"""Stable profile identities, separate from transient measurements."""

from dataclasses import dataclass, asdict
from vibeforge.storage.atomic import digest


@dataclass
class Profile:
    model: dict
    hardware: dict
    environment: dict
    workload: dict

    def to_dict(self):
        result = asdict(self)
        result["profile_hash"] = digest(result)
        return result
