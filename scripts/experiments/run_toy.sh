#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$project_root/scripts/ops/run.sh" "${1:-toy-validation}" --config configs/campaigns/toy.json
