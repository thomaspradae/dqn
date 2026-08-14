#!/usr/bin/env bash
set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
    SOURCE="$(readlink -f "$SOURCE")"
fi
SCRIPT_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
exec python3 "${SCRIPT_DIR}/eval_checkpoint.py" "$@"
