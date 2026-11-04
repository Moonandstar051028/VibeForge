"""Advisory locks held by a live supervisor or device user."""

import fcntl
from contextlib import contextmanager
from pathlib import Path


class LockBusy(RuntimeError):
    pass


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LockBusy(f"Another process holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
