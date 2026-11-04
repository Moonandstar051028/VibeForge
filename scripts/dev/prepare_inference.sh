#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/../.." && pwd)"
# Add the small missing adapter dependency to this project only.
# Torch/Transformers remain supplied by the configured inference interpreter.
python3 -m pip install --target "$project_root/var/cache/inference-packages" --no-deps "accelerate==1.15.0"
