import os
import subprocess
from pathlib import Path

import pytest

from vibeforge.storage.atomic import read_json


@pytest.mark.inference
def test_real_transformers_two_architectures_two_precisions(project, tmp_path):
    python = os.environ.get("VIBEFORGE_TEST_INFERENCE_PYTHON")
    if not python:
        pytest.skip("Set VIBEFORGE_TEST_INFERENCE_PYTHON to a local Transformers/PyTorch environment")
    env = dict(os.environ, PYTHONPATH=str(project / "src"), HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([python, str(project / "tests/fixtures/tiny_transformers_probe.py"),
                             "--project", str(project), "--output", str(tmp_path / "probe")],
                            env=env, text=True, capture_output=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    report = read_json(tmp_path / "probe/report.json")
    assert report["passed"] and len(report["results"]) == 4
