#!/usr/bin/env bash
set -euo pipefail

WATCHER_PATH="${WATCHER_PATH:-/home/uace/researchops/cluster_watch/check_nodes.sh}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [ ! -f "$WATCHER_PATH" ]; then
    echo "watcher not found: $WATCHER_PATH" >&2
    exit 1
fi

cp -a "$WATCHER_PATH" "${WATCHER_PATH}.bak.${STAMP}"

python3 - "$WATCHER_PATH" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()

replacement = r'''status_snapshot() {
  if [[ -s "${STATE_FILE}" ]]; then
    awk -F'\t' '{printf "%s %s %s %s\n", $1, $3, $4, $5}' "${STATE_FILE}"
  else
    uptime
    free -m | sed -n '1,2p'
  fi

  if command -v squeue >/dev/null 2>&1; then
    local queue
    queue="$(timeout 5s squeue -h -o '%i %P %j %T %M %R' 2>/dev/null | sed -n '1,8p')" || queue=""
    if [[ -n "${queue}" ]]; then
      printf '\nSlurm queue:\n%s\n' "${queue}"
    fi
  fi
}'''

pattern = re.compile(r"status_snapshot\(\) \{\n.*?\n\}", re.S)
new_text, count = pattern.subn(lambda _match: replacement, text, count=1)
if count != 1:
    raise SystemExit("could not replace status_snapshot()")

if "timeout 12s poormans" in new_text:
    raise SystemExit("bare poormans call still present in watcher")

path.write_text(new_text)
PY

chmod 700 "$WATCHER_PATH"
echo "patched $WATCHER_PATH"
echo "backup: ${WATCHER_PATH}.bak.${STAMP}"
echo "test with: bash -n $WATCHER_PATH && $WATCHER_PATH"
