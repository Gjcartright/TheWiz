from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".env.example"
TARGET = ROOT / ".env.local"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a private .env.local file for local API credentials.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing .env.local file.")
    args = parser.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f"missing template: {TEMPLATE}")
    if TARGET.exists() and not args.force:
        print(f"{TARGET} already exists; leaving it unchanged.")
        print("Edit that file in VS Code to add your API keys.")
        return

    TARGET.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"created {TARGET}")
    print("Open .env.local in VS Code and fill in your private Crypto Wizards and dYdX values.")


if __name__ == "__main__":
    main()
