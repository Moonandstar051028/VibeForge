import time

import pytest

from vibeforge.engine.client import EngineClient


def test_engine_client_rejects_requests_after_shared_phase_deadline():
    client = EngineClient(
        "http://127.0.0.1:1", timeout=60,
        deadline=time.monotonic() - 1, deadline_name="test phase",
    )

    with pytest.raises(TimeoutError, match="test phase deadline exceeded"):
        client.health()
