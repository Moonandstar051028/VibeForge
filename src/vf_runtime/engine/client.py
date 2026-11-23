"""HTTP client with explicit deadlines and bounded response sizes."""

import json
import math
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError


class EngineClient:
    def __init__(self, url, timeout=120, deadline=None, deadline_name="engine request"):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.deadline, self.deadline_name = deadline, deadline_name
        self.startup_seconds = None

    def bounded(self, deadline, name):
        """Return a client whose requests share one monotonic phase deadline."""
        result = EngineClient(self.url, self.timeout, deadline, name)
        result.startup_seconds = self.startup_seconds
        return result

    def request(self, operation, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(self.url + "/" + operation, data=data,
                          headers={"Content-Type": "application/json"})
        timeout = self.timeout
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.deadline_name} deadline exceeded")
            timeout = min(timeout, remaining)
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read(64 * 1024 * 1024 + 1)
                if len(body) > 64 * 1024 * 1024:
                    raise RuntimeError("Engine response exceeds limit")
                return json.loads(body)
        except HTTPError as error:
            raise RuntimeError(error.read(4096).decode(errors="replace")) from error

    def health(self):
        return self.request("health")

    def generate(self, **arguments):
        result = self.request("generate", arguments)
        if not isinstance(result.get("token_ids"), list) or not isinstance(result.get("input_ids"), list):
            raise ValueError("Engine omitted token IDs")
        for key in ("elapsed_seconds", "ttft_seconds", "tpot_seconds"):
            value = result.get(key)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid engine metric: {key}")
        return result

    def chat(self, **arguments):
        return self.request("chat", arguments)

    def logits(self, **arguments):
        return self.request("logits", arguments)

    def profile(self, **arguments):
        return self.request("profile", arguments)

    def implementation(self, **arguments):
        return self.request("implementation", arguments)

    def telemetry_start(self, **arguments):
        return self.request("telemetry/start", arguments)

    def telemetry_report(self, session_id):
        return self.request("telemetry/report", {"session_id": session_id})

    def telemetry_finish(self, session_id):
        return self.request("telemetry/finish", {"session_id": session_id})

    def tokenize(self, **arguments):
        result = self.request("tokenize", arguments)
        if not isinstance(result.get("count"), int) or result["count"] < 0:
            raise ValueError("Engine returned an invalid token count")
        return result
