#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/home/uace/researchops/cluster_watch}"
ENV_FILE="${BASE}/telegram.env"
STATE_FILE="${BASE}/state.tsv"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "missing ${ENV_FILE}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
  echo "telegram env missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID" >&2
  exit 2
fi

snapshot="$(
  if [[ -s "${STATE_FILE}" ]]; then
    awk -F'\t' '{printf "%s %s %s %s\n", $1, $3, $4, $5}' "${STATE_FILE}"
  else
    uptime
    free -m | sed -n '1,2p'
  fi
)"

queue="$(timeout 5s squeue -h -o '%i %P %j %T %M %R' 2>/dev/null | sed -n '1,8p' || true)"
if [[ -n "${queue}" ]]; then
  snapshot="${snapshot}

Slurm queue:
${queue}"
fi

msg="Telegram watcher snapshot test on $(hostname)
$(date -u '+%Y-%m-%d %H:%M:%S UTC')

This should be compact plain text, with no poormans ASCII dashboard.

Snapshot:
${snapshot}"

response_file="$(mktemp)"
http_code="$(
  curl -sS -o "${response_file}" -w '%{http_code}' \
    -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=${msg}"
)"

python3 - "${http_code}" "${response_file}" <<'PY'
import json
import sys
from pathlib import Path

http_code = sys.argv[1]
path = Path(sys.argv[2])
try:
    data = json.loads(path.read_text())
except Exception as exc:
    print(f"telegram http={http_code} invalid_json={exc}")
    raise SystemExit(1)

message_id = (data.get("result") or {}).get("message_id")
print(f"telegram http={http_code} ok={data.get('ok')} message_id={message_id}")
if not data.get("ok"):
    print(f"description={data.get('description')}")
    raise SystemExit(1)
PY

rm -f "${response_file}"
