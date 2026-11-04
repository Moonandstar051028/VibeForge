"""Record the supervisor and inference environment independently."""

import json
import os
import platform
import subprocess
import sys


def inspect_environment(python=None, extra_paths=()):
    result = {"controller_python": sys.version, "platform": platform.platform()}
    if python:
        script = (
            "import importlib.metadata as m,json,sys;"
            "names=['torch','transformers','safetensors','accelerate','bitsandbytes'];"
            "found={p.metadata['Name'].lower():p.version for p in m.distributions()};"
            "print(json.dumps({'python':sys.version,'packages':{n:found.get(n) for n in names}}))"
        )
        try:
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(extra_paths) + os.pathsep + environment.get("PYTHONPATH", "")
            process = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=20, env=environment)
            result["inference"] = json.loads(process.stdout) if process.returncode == 0 else {"error": process.stderr[-2000:]}
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            result["inference"] = {"error": str(error)}
    return result
