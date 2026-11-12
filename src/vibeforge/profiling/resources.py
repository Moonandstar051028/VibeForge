"""Controller-owned continuous host and device resource sampling."""

import threading
from collections import defaultdict
from pathlib import Path

from vibeforge.profiling.hardware import gpu_telemetry


def descendants(pids):
    found, pending = set(), list(pids)
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        try:
            pending.extend(map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split()))
        except (OSError, ValueError):
            pass
    return found


def rss_bytes(pids):
    values = []
    for pid in descendants(pids):
        try:
            values.extend(int(line.split()[1]) * 1024 for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                          if line.startswith("VmRSS:"))
        except (OSError, ValueError, IndexError):
            pass
    return sum(values) if values else None


class ResourceSampler:
    def __init__(self, pids, interval=0.05):
        self.pids, self.interval = tuple(pids), interval
        self.peak, self.samples = None, 0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        value = rss_bytes(self.pids)
        if value is not None:
            self.peak = max(self.peak or 0, value)
            self.samples += 1

    def _run(self):
        while not self.stopped.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join(timeout=2)
        self._sample()


class DeviceTelemetrySampler:
    """Continuously aggregate typed telemetry for CUDA-visible devices."""

    METRICS = (
        "temperature_c", "gpu_utilization_pct", "memory_utilization_pct",
        "sm_clock_mhz", "memory_clock_mhz", "power_watts", "power_limit_watts",
    )

    def __init__(self, interval=1.0, enabled=True):
        self.interval, self.enabled = interval, enabled
        self.sample_rounds = self.available_rounds = 0
        self.observations = defaultdict(list)
        self.identities = {}
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        if not self.enabled:
            return
        rows = gpu_telemetry()
        self.sample_rounds += 1
        if rows is None:
            return
        self.available_rounds += 1
        for row in rows:
            key = (row.get("visible_index"), row.get("uuid"))
            self.identities[key] = {
                name: row.get(name) for name in ("visible_index", "physical_index", "uuid")
            }
            for name in self.METRICS:
                value = row.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.observations[(key, name)].append(float(value))

    def _run(self):
        while not self.stopped.wait(self.interval):
            self._sample()

    def __enter__(self):
        if self.enabled:
            self._sample()
            self.thread.start()
        return self

    def __exit__(self, *_):
        if self.enabled:
            self.stopped.set()
            self.thread.join(timeout=max(2, self.interval * 2))
            self._sample()

    def report(self):
        devices = []
        for key, identity in sorted(self.identities.items(), key=lambda item: item[0][0]):
            metrics = {}
            for name in self.METRICS:
                values = self.observations.get((key, name), [])
                if values:
                    metrics[name] = {
                        "samples": len(values), "mean": sum(values) / len(values),
                        "minimum": min(values), "maximum": max(values),
                    }
            devices.append({**identity, "metrics": metrics})
        status = "disabled" if not self.enabled else (
            "collected" if devices else "unavailable")
        return {
            "status": status, "interval_seconds": self.interval,
            "sample_rounds": self.sample_rounds,
            "available_rounds": self.available_rounds,
            "devices": devices,
            "scope": (
                "continuous NVIDIA management telemetry for CUDA-visible devices; "
                "utilization.memory is not memory bandwidth and power is not integrated energy"
            ),
        }
