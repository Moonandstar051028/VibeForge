"""Event creation with wall-clock and within-boot monotonic timestamps."""

import datetime
import time
import uuid
from pathlib import Path


def boot_id():
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text().strip() if path.exists() else "unknown"


def make_event(campaign_id, event_type, **fields):
    return {
        "event_id": uuid.uuid4().hex,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "monotonic_time": time.monotonic(),
        "boot_id": boot_id(),
        "campaign_id": campaign_id,
        "event_type": event_type,
        **fields,
    }
