from __future__ import annotations

import argparse
import json

from quant_platform.udemy_transcript_vault import build_udemy_transcript_vault_inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recheck-unavailable", action="store_true")
    args = parser.parse_args()
    result = build_udemy_transcript_vault_inventory(recheck_unavailable=args.recheck_unavailable)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
