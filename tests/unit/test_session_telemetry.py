from vibeforge.engine.session_telemetry import SessionTelemetry


def test_session_telemetry_aggregates_requests_and_enforces_bound():
    telemetry = SessionTelemetry()
    telemetry.start("agent", max_requests=2, retention_seconds=60)
    telemetry.record({
        "operation": "chat_completion", "protocol": "responses",
        "backend_wall_seconds": 2.0, "queue_seconds": 0.2,
        "input_tokens": 10, "output_tokens": 4, "ttft_seconds": 0.5,
        "tpot_seconds": 0.25, "phases": {"decode_forward_seconds": 1.0},
    })
    telemetry.record({
        "operation": "chat_completion", "protocol": "responses",
        "backend_wall_seconds": 1.0, "queue_seconds": 0.1,
        "input_tokens": 6, "output_tokens": 2, "ttft_seconds": 0.25,
        "tpot_seconds": 0.2, "phases": {"decode_forward_seconds": 0.4},
    })
    telemetry.record({"operation": "generate", "backend_wall_seconds": 99.0})

    report = telemetry.finish("agent")

    assert report["request_count"] == 2 and report["dropped_requests"] == 1
    assert report["operations"] == {"chat_completion": 2}
    assert report["input_tokens"] == 16 and report["output_tokens"] == 6
    assert report["backend_wall_seconds"] == 3.0
    assert report["aggregate_output_tokens_per_second"] == 2.0
    assert report["latency_seconds"] == {"p50": 1.0, "p95": 2.0, "max": 2.0}
