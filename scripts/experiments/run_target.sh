#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$project_root/scripts/ops/run.sh" "${1:-qwen36-validation}" --config configs/campaigns/qwen36-35b.json
