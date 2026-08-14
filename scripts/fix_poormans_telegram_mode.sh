#!/usr/bin/env bash
set -euo pipefail

POORMANS_PATH="${POORMANS_PATH:-/home/uace/.local/bin/poormans}"
WATCHER_PATH="${WATCHER_PATH:-/home/uace/researchops/cluster_watch/check_nodes.sh}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [ ! -f "$POORMANS_PATH" ]; then
    echo "poormans not found: $POORMANS_PATH" >&2
    exit 1
fi

cp -a "$POORMANS_PATH" "${POORMANS_PATH}.bak.${STAMP}"

python3 - "$POORMANS_PATH" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

if "TELEGRAM_MODE=0" not in text:
    text = text.replace(
        'POORMANS_VERSION="0.1.1"\n\ncase "${1:-}" in\n',
        'POORMANS_VERSION="0.1.1"\nTELEGRAM_MODE=0\n\ncase "${1:-}" in\n',
        1,
    )

text = text.replace(
    '    --telegram)\n        shift\n        ;;\n',
    '    --telegram)\n        TELEGRAM_MODE=1\n        shift\n        ;;\n',
    1,
)

if "print_telegram_summary()" not in text:
    marker = '# ── Main ─────────────────────────────────────────────────\n'
    helper = r'''
# ── Telegram-safe summary ─────────────────────────────────
print_telegram_summary() {
    local prom_state="down"
    prom_ok && prom_state="ok"
    local prom_host="${PROM#http://}"
    prom_host="${prom_host#https://}"

    printf 'Cluster snapshot\n'
    printf 'prometheus: %s (%s)\n' "$prom_state" "$prom_host"

    local node label instance ssh_target location ssh_state metrics_state up_value slurm_node jobs job_state
    for node in "${NODES[@]}"; do
        IFS='|' read -r label instance ssh_target location <<< "$node"

        ssh_state="down"
        ssh_up "$ssh_target" && ssh_state="up"

        metrics_state="unknown"
        if prom_ok; then
            up_value=$(query "up{instance=\"$instance\"}")
            case "$up_value" in
                1|1.0) metrics_state="up" ;;
                0|0.0) metrics_state="down" ;;
                *) metrics_state="unknown" ;;
            esac
        fi

        slurm_node=$(slurm_node_for_label "$label")
        jobs=$(get_node_jobs "$slurm_node" | tr '\n' '; ' | sed 's/[;[:space:]]*$//')
        if [ -n "$jobs" ]; then
            job_state=" jobs=${jobs}"
        else
            job_state=""
        fi

        printf '%s: ssh=%s metrics=%s%s\n' "$label" "$ssh_state" "$metrics_state" "$job_state"
    done
}

'''
    if marker not in text:
        raise SystemExit("could not find main marker in poormans script")
    text = text.replace(marker, helper + marker, 1)

main_old = '''choose_prom >/dev/null 2>&1 || true
T0=$(date +%s%3N)
print_header
'''
main_new = '''choose_prom >/dev/null 2>&1 || true
if [ "${TELEGRAM_MODE:-0}" = "1" ]; then
    print_telegram_summary
    exit 0
fi
T0=$(date +%s%3N)
print_header
'''
if main_new not in text:
    if main_old not in text:
        raise SystemExit("could not find poormans main entrypoint")
    text = text.replace(main_old, main_new, 1)

path.write_text(text)
PY

chmod 755 "$POORMANS_PATH"

if [ -f "$WATCHER_PATH" ]; then
    cp -a "$WATCHER_PATH" "${WATCHER_PATH}.bak.${STAMP}"
    python3 - "$WATCHER_PATH" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
before = text

# Keep explicit poormans management commands alone. Only adjust snapshot-style
# command substitutions or timeout calls that invoke the display dashboard.
patterns = [
    (r'(\$\([^)\n]*\bpoormans)(\s*2>)', r'\1 --telegram\2'),
    (r'(\btimeout\s+[0-9]+s?\s+poormans)(\s*2>)', r'\1 --telegram\2'),
    (r'(\btimeout\s+[0-9]+s?\s+poormans)(\s*\|)', r'\1 --telegram\2'),
    (r'(\btimeout\s+[0-9]+s?\s+poormans)(\s*$)', r'\1 --telegram\2'),
]
for pattern, replacement in patterns:
    text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

if text != before:
    path.write_text(text)
    print(f"updated watcher snapshot calls in {path}")
else:
    print(f"watcher unchanged: inspect {path} and ensure alert snapshots call 'poormans --telegram', not bare 'poormans'")
PY
fi

echo "patched $POORMANS_PATH"
echo "test with: $POORMANS_PATH --telegram"
