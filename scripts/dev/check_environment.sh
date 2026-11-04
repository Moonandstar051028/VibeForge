#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
"${VIBEFORGE_PYTHON:-$project_root/.venv/bin/python}" -m vibeforge profile --config configs/campaigns/qwen36-35b.json
