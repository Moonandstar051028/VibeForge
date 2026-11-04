"""Controlled quality-gate fixture; scores are deliberate, not model-quality claims."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--reject-float32", action="store_true")
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
with urlopen(request["engine_url"] + "/health", timeout=5) as response:
    health = json.load(response)
score = .5 if args.reject_float32 and health["precision"]["compute_dtype"] == "float32" else 1.0
Path(args.output).write_text(json.dumps({"passed": True, "metrics": {"accuracy": score}}))
