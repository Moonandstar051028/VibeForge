#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
"${VIBEFORGE_PYTHON:-python3}" -m pytest -m "not target" "$@"
