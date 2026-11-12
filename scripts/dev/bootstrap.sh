#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -x "$project_root/.venv/bin/python" ]]; then
  "${VIBEFORGE_BOOTSTRAP_PYTHON:-python3}" -m venv --system-site-packages "$project_root/.venv"
fi
"$project_root/.venv/bin/python" -m pip install -e "$project_root"
