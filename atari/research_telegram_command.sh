#!/usr/bin/env bash
set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
    SOURCE="$(readlink -f "$SOURCE")"
fi
SCRIPT_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
cmd="${1:-status}"
[ "$#" -gt 0 ] && shift || true

case "$cmd" in
    /status|status)
        exec "${SCRIPT_DIR}/research_status.sh" status "$@"
        ;;
    /resources|resources)
        exec "${SCRIPT_DIR}/research_status.sh" resources "$@"
        ;;
    /runs|runs)
        exec "${SCRIPT_DIR}/research_status.sh" runs "$@"
        ;;
    /tail|tail)
        exec "${SCRIPT_DIR}/research_status.sh" tail "$@"
        ;;
    /best|best)
        exec "${SCRIPT_DIR}/research_status.sh" best "$@"
        ;;
    /archive|archive)
        exec "${SCRIPT_DIR}/research_status.sh" archive "$@"
        ;;
    /sb3|sb3)
        exec "${SCRIPT_DIR}/research_status.sh" sb3 "$@"
        ;;
    /report|report)
        exec "${SCRIPT_DIR}/research_status.sh" report "$@"
        ;;
    /help|help)
        exec "${SCRIPT_DIR}/research_status.sh" help "$@"
        ;;
    /eval|eval)
        exec "${SCRIPT_DIR}/eval_checkpoint.sh" "$@"
        ;;
    /queue|queue)
        if [ "${1:-}" = "drain" ]; then
            shift
            exec "${SCRIPT_DIR}/eval_checkpoint.sh" drain "$@"
        fi
        exec "${SCRIPT_DIR}/eval_checkpoint.sh" queue "$@"
        ;;
    /auto-eval|auto-eval|/autoeval|autoeval)
        exec "${SCRIPT_DIR}/auto_eval_milestones.sh" "$@"
        ;;
    *)
        printf 'unknown command: %s\n' "$cmd" >&2
        printf 'commands: status resources runs tail best sb3 report eval queue auto-eval help\n' >&2
        exit 2
        ;;
esac
