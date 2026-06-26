#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_FILE="${1:-.env.local}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE"
  exit 1
fi

CONFIG_JSON="$(PYTHONPATH=src ENV_FILE="$ENV_FILE" python3 - <<'PY'
import json
import os

from quant_platform.env import load_env_file
from quant_platform.api_extraction import CryptoWizardsExtractor, CryptoWizardsLiveConfig

load_env_file(os.environ["ENV_FILE"])
config = CryptoWizardsLiveConfig.from_env()
missing = config.missing_requirements()
first = config.endpoints[0] if config.endpoints else None
extractor = CryptoWizardsExtractor.from_live_config(config) if not missing else None
print(json.dumps({
    "missing": missing,
    "api_key_present": bool(config.api_key),
    "api_key": config.api_key or "",
    "endpoint_name": first.name if first else "",
    "url": extractor.endpoint_url(first) if extractor and first else "",
}))
PY
)"

MISSING="$(CONFIG_JSON="$CONFIG_JSON" python3 - <<'PY'
import json, os
print(",".join(json.loads(os.environ["CONFIG_JSON"])["missing"]))
PY
)"

if [[ -n "$MISSING" ]]; then
  echo "Crypto Wizards config missing: $MISSING"
  exit 1
fi

API_KEY_PRESENT="$(CONFIG_JSON="$CONFIG_JSON" python3 - <<'PY'
import json, os
print("yes" if json.loads(os.environ["CONFIG_JSON"])["api_key_present"] else "no")
PY
)"
if [[ "$API_KEY_PRESENT" != "yes" ]]; then
  echo "Crypto Wizards API key missing"
  exit 1
fi

ENDPOINT_NAME="$(CONFIG_JSON="$CONFIG_JSON" python3 - <<'PY'
import json, os
print(json.loads(os.environ["CONFIG_JSON"])["endpoint_name"])
PY
)"
URL="$(CONFIG_JSON="$CONFIG_JSON" python3 - <<'PY'
import json, os
print(json.loads(os.environ["CONFIG_JSON"])["url"])
PY
)"
API_KEY="$(CONFIG_JSON="$CONFIG_JSON" python3 - <<'PY'
import json, os
print(json.loads(os.environ["CONFIG_JSON"])["api_key"])
PY
)"

OUTPUT="data/raw/${ENDPOINT_NAME}.json"
HEADER_FILE="$(mktemp)"
trap 'rm -f "$HEADER_FILE"' EXIT
chmod 600 "$HEADER_FILE"
{
  printf 'X-api-key: %s\n' "$API_KEY"
  printf 'Content-Type: application/json\n'
} > "$HEADER_FILE"

mkdir -p data/raw docs

echo "curling Crypto Wizards endpoint"
echo "endpoint: $ENDPOINT_NAME"
echo "url: $URL"
echo "output: $OUTPUT"

curl --fail --show-error --silent \
  --connect-timeout 20 \
  --max-time 60 \
  --header "@${HEADER_FILE}" \
  "$URL" \
  --output "$OUTPUT"

PYTHONPATH=src python3 - <<'PY'
import json
from pathlib import Path

from quant_platform.api_extraction import CryptoWizardsExtractor

payloads = {}
for path in Path("data/raw").glob("*.json"):
    if path.name.startswith("crypto_wizards_pair_metrics_sample"):
        continue
    try:
        payloads[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue

if not payloads:
    raise SystemExit("no valid live Crypto Wizards JSON payloads found in data/raw")

output = Path("docs/crypto_wizards_live_field_dictionary.csv")
CryptoWizardsExtractor.write_discovered_fields(payloads, output)
print(f"field_dictionary: {output}")
print(f"payloads: {len(payloads)}")
PY
