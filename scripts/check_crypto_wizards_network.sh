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
from urllib.parse import urlparse

from quant_platform.env import load_env_file
from quant_platform.api_extraction import CryptoWizardsLiveConfig, CryptoWizardsExtractor

load_env_file(os.environ["ENV_FILE"])
config = CryptoWizardsLiveConfig.from_env()
missing = config.missing_requirements()
first = config.endpoints[0] if config.endpoints else None
extractor = CryptoWizardsExtractor.from_live_config(config) if not missing else None
url = extractor.endpoint_url(first) if extractor and first else ""
print(json.dumps({
    "missing": missing,
    "base_url_present": bool(config.base_url),
    "api_key_present": bool(config.api_key),
    "endpoint_count": len(config.endpoints),
    "endpoint_name": first.name if first else "",
    "host": urlparse(config.base_url or "").hostname or "",
    "url": url,
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
  CONFIG_JSON="$CONFIG_JSON" python3 - <<'PY'
import json, os
config = json.loads(os.environ["CONFIG_JSON"])
print(f"CRYPTO_WIZARDS_BASE_URL present: {'yes' if config['base_url_present'] else 'no'}")
print(f"CRYPTO_WIZARDS_ENDPOINTS present: {'yes' if config['endpoint_count'] else 'no'}")
PY
  exit 1
fi

HOST="$(CONFIG_JSON="$CONFIG_JSON" python3 - <<'PY'
import json, os
print(json.loads(os.environ["CONFIG_JSON"])["host"])
PY
)"
FULL_URL="$(CONFIG_JSON="$CONFIG_JSON" python3 - <<'PY'
import json, os
print(json.loads(os.environ["CONFIG_JSON"])["url"])
PY
)"

echo "Crypto Wizards network check"
echo "env_file: $ENV_FILE"
CONFIG_JSON="$CONFIG_JSON" python3 - <<'PY'
import json, os
config = json.loads(os.environ["CONFIG_JSON"])
print(f"base_url_present: {'yes' if config['base_url_present'] else 'no'}")
print(f"api_key_present: {'yes' if config['api_key_present'] else 'no'}")
print(f"endpoint_count: {config['endpoint_count']}")
print(f"endpoint: {config['endpoint_name']}")
PY
echo "host: $HOST"
echo "url: $FULL_URL"
echo

echo "Python DNS check"
HOST="$HOST" python3 - <<'PY'
import socket

import os

host = os.environ["HOST"]
try:
    rows = socket.getaddrinfo(host, 443)
    addresses = sorted({row[4][0] for row in rows})
    print(f"dns_ok: true")
    print(f"addresses: {addresses}")
except Exception as exc:
    print("dns_ok: false")
    print(f"dns_error: {type(exc).__name__}: {exc}")
PY
echo

echo "curl HEAD check"
curl -I --max-time 20 "$FULL_URL" || true
echo

echo "platform diagnostic"
PYTHONPATH=src python3 -m quant_platform.cli diagnose-crypto-wizards || true
echo

echo "platform crawl attempt"
PYTHONPATH=src python3 -m quant_platform.cli crawl-crypto-wizards || true
