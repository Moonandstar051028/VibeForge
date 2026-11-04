#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
"${VIBEFORGE_PYTHON:-python3}" -m vibeforge profile --config configs/campaigns/qwen36-35b.json
