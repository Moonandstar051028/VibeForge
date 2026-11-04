"""Strict scalar metric collection without invented device measurements."""

import math
from vibeforge.observability.logs import append_record


def record_metric(path, name, value, **labels):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid metric {name}: {value}")
    append_record(path, {"name": name, "value": value, "labels": labels})
