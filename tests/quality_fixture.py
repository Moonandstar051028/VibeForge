"""Independent, real-HTTP quality fixture used only in controller tests."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument('--request')
parser.add_argument('--output')
parser.add_argument('--expected')
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
payload = json.dumps({'prompt': 'quality-only', 'max_new_tokens': 2}).encode()
with urlopen(Request(request['engine_url'] + '/generate', data=payload,
                     headers={'Content-Type': 'application/json'}), timeout=5) as response:
    result = json.load(response)
score = float(result['token_ids'] == json.loads(args.expected))
Path(args.output).write_text(json.dumps({'passed': True, 'metrics': {'accuracy': score},
                                      'cases': [{'id': 'quality-only', 'passed': bool(score)}]}))
