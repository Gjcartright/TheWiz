#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER_PATH="$ROOT_DIR/scripts/capture_crypto_wizards_pair_detail.js"

if [[ ! -f "$HELPER_PATH" ]]; then
  echo "Capture helper not found: $HELPER_PATH" >&2
  exit 1
fi

if ! command -v pbcopy >/dev/null 2>&1; then
  echo "pbcopy is not available. Open this file and copy it manually:" >&2
  echo "$HELPER_PATH" >&2
  exit 1
fi

if ! pbcopy < "$HELPER_PATH"; then
  echo "Could not write to the macOS clipboard from this shell." >&2
  echo "Open this file in VS Code and copy its contents manually:" >&2
  echo "$HELPER_PATH" >&2
  exit 1
fi

cat <<EOF
Crypto Wizards capture helper copied to clipboard.

Next:
1. Open the authenticated pair page:
   https://cryptowizards.net/wizards/zscore/pair/1

2. Open the browser console and paste the helper.

3. Click the pair page refresh/recalculate icon.

4. Run:
   await __CW_CAPTURE_STATUS__()

5. If useful payloads are captured, run:
   await __CW_DOWNLOAD_CAPTURE__()

6. Leave the downloaded JSON in your Downloads folder.

7. Import and verify it:
   PYTHONPATH=src python -m quant_platform.cli import-latest-pair-detail-download
   PYTHONPATH=src python -m quant_platform.cli priority-dashboard
EOF
