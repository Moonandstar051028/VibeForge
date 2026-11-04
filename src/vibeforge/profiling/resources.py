"""Controller-owned process-tree RSS sampling, separate from device memory."""

import threading
from pathlib import Path


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
