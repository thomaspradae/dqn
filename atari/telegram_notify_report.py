#!/usr/bin/env python3
"""Send an auto-evaluation report through the existing DQN Telegram bot config."""

import argparse
import os
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_CONFIG = Path.home() / "cluster" / "alerts" / "config.env"
MAX_TELEGRAM_TEXT = 3900


def load_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("DQN_TELEGRAM_CONFIG", DEFAULT_CONFIG)),
    )
    args = parser.parse_args()

    if not args.summary.is_file():
        raise SystemExit(f"summary file is missing: {args.summary}")
    if not args.config.is_file():
        raise SystemExit(f"Telegram config is missing: {args.config}")

    config = load_config(args.config)
    token = config.get("BOT_TOKEN")
    chat_id = config.get("CHAT_ID")
    if not token or not chat_id:
        raise SystemExit("Telegram config must define BOT_TOKEN and CHAT_ID")

    text = args.summary.read_text().strip()
    if len(text) > MAX_TELEGRAM_TEXT:
        text = text[: MAX_TELEGRAM_TEXT - 80].rstrip() + "\n... truncated ..."
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        if not (200 <= response.status < 300):
            raise SystemExit(f"Telegram send failed with HTTP {response.status}")
    print("Telegram report sent")


if __name__ == "__main__":
    main()
