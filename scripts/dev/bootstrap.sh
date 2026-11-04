#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 -m venv --system-site-packages "$project_root/.venv"
"$project_root/.venv/bin/python" -m pip install --no-build-isolation --no-deps -e "$project_root"
