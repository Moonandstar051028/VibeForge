"""HTTP client with explicit deadlines and bounded response sizes."""

import json
import math
from urllib.request import Request, urlopen
from urllib.error import HTTPError


class EngineClient:
    def __init__(self, url, timeout=120):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.startup_seconds = None

    def request(self, operation, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(self.url + "/" + operation, data=data,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
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
