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
            "found={}\n"
            "for p in m.distributions():\n"
            " name=p.metadata.get('Name','').lower()\n"
            " if name: found.setdefault(name,p.version)\n"
            "result={'python':sys.version,'packages':{**found,**{n:found.get(n) for n in names}}};\n"
            "try:\n"
            " import torch\n"
            " runtime={'cpu':True,'backend':'rocm' if torch.version.hip else 'cuda','devices':[]}\n"
            " if torch.cuda.is_available():\n"
            "  for i in range(torch.cuda.device_count()):\n"
            "   p=torch.cuda.get_device_properties(i); free,total=torch.cuda.mem_get_info(i)\n"
            "   runtime['devices'].append({'index':i,'name':p.name,'total_bytes':int(total),'free_bytes':int(free),"
            "'bf16_supported':bool(torch.cuda.is_bf16_supported()),'capability':list(torch.cuda.get_device_capability(i))})\n"
            " result['runtime']=runtime\n"
            "except Exception as e: result['runtime']={'probe_error':str(e)}\n"
            "print(json.dumps(result))"
        )
        try:
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(extra_paths) + os.pathsep + environment.get("PYTHONPATH", "")
            process = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=20, env=environment)
            result["inference"] = json.loads(process.stdout) if process.returncode == 0 else {"error": process.stderr[-2000:]}
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            result["inference"] = {"error": str(error)}
    return result
