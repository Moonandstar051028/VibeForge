import json
import sys
from vibeforge.observability.logs import run_command


def test_timeout_terminates_and_audits_owned_process(tmp_path):
    record = tmp_path / "process.json"
    trace = tmp_path / "commands.jsonl"
    result = run_command([sys.executable, "-c", "import time; time.sleep(20)", "test-owner-token"],
                         tmp_path, tmp_path / "output.log", timeout=0.15, trace=trace,
                         process_record=record, token="test-owner-token")
    assert result["exit_code"] == 124
    assert not record.exists()
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [row["status"] for row in events] == ["started", "completed"]
