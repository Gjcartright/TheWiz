"""Verify the pair-page input capture redaction source decision."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
RELATIVE = "scripts/capture_crypto_wizards_pair_detail_ui_bundle.js"
JUNIT = Path(
    "/Users/gregc/Backups/TheWiz/pair-capture-diagnostics/2026-09-30/pair-capture-junit.xml"
)
OUTPUT = AUDIT / "GATE0_PAIR_CAPTURE_REDACTION_DECISION_2026-09-30.json"

MOCK_BROWSER_CHECK = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
let source = fs.readFileSync(process.argv[2], 'utf8');
const marker = '  window.__CW_ORIENTATION_STATUS__ =';
assert.equal(source.split(marker).length, 2);
source = source.replace(marker, '  window.__TEST_CAPTURE_STATE__ = captureState;\n' + marker);
const input = (type, value, autocomplete = '') => ({
  type, value, autocomplete, name: '', min: '', max: '', step: '',
  parentElement: { innerText: '' },
});
const inputs = [
  input('password', 'password-placeholder'),
  input('email', 'person@example.com'),
  input('text', 'api_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456'),
  input('text', '123456', 'one-time-code'),
];
const window = {};
const document = {
  querySelectorAll: (selector) => selector === 'input' ? inputs : [],
  title: 'fixture', body: { innerText: '' },
};
vm.runInNewContext(source, {
  window, document, location: { href: 'https://example.invalid/pair' },
  console: { log() {} }, URL, Date, setTimeout,
});
const values = Array.from(window.__TEST_CAPTURE_STATE__().inputs, row => row.value);
assert.deepStrictEqual(values, [
  '[REDACTED]', '[REDACTED]', '[REDACTED_TOKEN]', '[REDACTED]',
]);
console.log('PASS mock browser input redaction');
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = [row for row in csv.DictReader(stream) if row["relative_path"] == RELATIVE]
    if len(rows) != 1 or rows[0]["custody_status"] != "REVIEWED_PORTED_RUNTIME_REDACTION_FIX":
        raise ValueError("capture source decision absent or duplicated")
    row = rows[0]
    current = ROOT / RELATIVE
    candidate = RUNTIME / RELATIVE
    if digest(current) != digest(candidate) or digest(candidate) != row["runtime_sha256_at_freeze"]:
        raise ValueError("capture source does not match the preserved candidate fix")
    if digest(current) == row["working_sha256_at_freeze"]:
        raise ValueError("active raw-value capture was not changed")
    source = current.read_text(encoding="utf-8")
    inputs = source.split('const inputs = Array.from(document.querySelectorAll("input"))', 1)[1]
    inputs = inputs.split("const stationarity =", 1)[0]
    if "value: safeInputValue(element)" not in inputs or "value: element.value" in inputs:
        raise ValueError("input serialization bypasses redaction")
    subprocess.run(["node", "--check", str(current)], check=True, capture_output=True)
    subprocess.run(
        ["node", "-", str(current)],
        input=MOCK_BROWSER_CHECK,
        text=True,
        check=True,
        capture_output=True,
    )
    junit = next(ET.parse(JUNIT).getroot().iter("testsuite"))
    if any(junit.attrib[key] != value for key, value in (("tests", "16"), ("failures", "0"), ("errors", "0"))):
        raise ValueError("focused pair capture tests did not pass")
    summary = {
        "schema_version": "thewiz.gate0.pair_capture_redaction_decision.v1",
        "decision": row["custody_status"],
        "source_path": RELATIVE,
        "frozen_active_sha256": row["working_sha256_at_freeze"],
        "preserved_runtime_sha256": row["runtime_sha256_at_freeze"],
        "current_sha256": digest(current),
        "javascript_syntax": "PASS",
        "mock_browser_input_redaction": "PASS_PASSWORD_EMAIL_TOKEN_ONE_TIME_CODE",
        "focused_python_tests_passed": 16,
        "focused_junit_sha256": digest(JUNIT),
        "interpretation": "The active browser capture now applies its existing sensitive-value redaction helper to input rows. No credential, network, order, or trading action was run.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS pair_capture_input_redaction")


if __name__ == "__main__":
    main()
