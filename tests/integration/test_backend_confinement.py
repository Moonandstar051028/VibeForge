import os
import subprocess
import sys


def test_generated_process_cannot_write_controller_files(tmp_path, project):
    cache = tmp_path / "cache"
    cache.mkdir()
    protected = tmp_path / "controller-state.json"
    protected.write_text("stable")
    script = """
from pathlib import Path
from vibeforge.engine.sandbox import restrict_writes
import sys
restrict_writes([sys.argv[1]])
with open('/dev/null', 'r+b') as sink:
    sink.write(b'library initialization probe')
Path(sys.argv[1], "allowed.txt").write_text("ok")
try:
    Path(sys.argv[2]).write_text("corrupt")
except PermissionError:
    print("write denied")
else:
    raise SystemExit(1)
"""
    result = subprocess.run([sys.executable, "-c", script, str(cache), str(protected)],
                            env=dict(os.environ, PYTHONPATH=str(project / "src")),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert protected.read_text() == "stable"
    assert (cache / "allowed.txt").read_text() == "ok"
