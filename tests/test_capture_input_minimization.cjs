"use strict";

// Offline fake-DOM tests of the reviewed pure capture projection only. Never
// evaluate the installing IIFE, authentication checks, selectors, or downloader.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const sourcePath = path.resolve(__dirname, "../scripts/capture_crypto_wizards_pair_detail_ui_bundle.js");
const source = fs.readFileSync(sourcePath, "utf8");
function section(start, end) {
  const begin = source.indexOf(start);
  const finish = source.indexOf(end, begin + start.length);
  assert.ok(begin >= 0 && finish > begin, `Reviewed projection marker missing: ${start}`);
  return source.slice(begin, finish);
}
const helpers = section("  const lower =", "  const memberNavigationTargets =");
const capture = section("  const captureState =", "  const chartOptions =");

function project(inputs) {
  const selectors = {
    input: inputs,
    select: [{
      type: "select-one", value: "3-1", selectedIndex: 0,
      options: [{ value: "3-1", text: "Static (Spread)", selected: true }],
    }],
    div: [], svg: [], button: [],
  };
  const document = {
    title: "Synthetic scientific fixture",
    body: { innerText: "BTC (asset X)\nETH (asset Y)\nHedge ratio 1.25" },
    querySelectorAll(selector) {
      assert.ok(Object.hasOwn(selectors, selector), `Unexpected selector: ${selector}`);
      return selectors[selector];
    },
  };
  return vm.runInNewContext(
    `${helpers}\n${capture}\ncaptureState();`,
    { document, location: { href: "https://example.invalid/scientific-fixture" } },
    { filename: "reviewed-pure-capture-projection", timeout: 1000 },
  );
}

function input(value, extra = {}) {
  return {
    type: "text", autocomplete: "", value, name: "scientific_fixture",
    min: "-5", max: "5", step: "0.01", parentElement: { innerText: "Entry threshold" },
    ...extra,
  };
}

for (const type of ["password", "email", "hidden"]) {
  test(`sensitive input type ${type} uses existing redaction`, () => {
    assert.equal(project([input("synthetic-sensitive-value", { type })]).inputs[0].value, "[REDACTED]");
  });
}

for (const autocomplete of ["current-password", "new-password", "one-time-code"]) {
  test(`sensitive autocomplete ${autocomplete} uses existing redaction`, () => {
    assert.equal(project([input("synthetic-sensitive-value", { autocomplete })]).inputs[0].value, "[REDACTED]");
  });
}

for (const [value, expected] of [
  ["reviewer@example.test", "[REDACTED_EMAIL]"],
  ["api_" + "a".repeat(20), "[REDACTED_TOKEN]"],
  ["0x" + "a".repeat(64), "[REDACTED_PRIVATE_VALUE]"],
]) {
  test(`ordinary text input applies existing ${expected} rule`, () => {
    assert.equal(project([input(value)]).inputs[0].value, expected);
  });
}

for (const value of ["0", "-2.5", "1e-5", "false", "BTCUSDT", "u1_given_u2"]) {
  test(`scientific input ${value} is preserved`, () => {
    assert.equal(project([input(value)]).inputs[0].value, value);
  });
}

test("input order, scientific metadata, selection and source DOM remain unchanged", () => {
  const elements = [input("-2.5"), input("1.25", { type: "number", name: "hedge_ratio" })];
  const before = JSON.stringify(elements);
  const result = project(elements);
  assert.equal(JSON.stringify(elements), before);
  assert.equal(result.inputs.length, 2);
  for (const [index, item] of result.inputs.entries()) {
    assert.equal(item.index, index);
    assert.equal(item.type, elements[index].type);
    assert.equal(item.name, elements[index].name);
    assert.equal(item.min, "-5");
    assert.equal(item.max, "5");
    assert.equal(item.step, "0.01");
    assert.equal(item.context, "Entry threshold");
  }
  assert.equal(result.selects[0].value, "3-1");
  assert.equal(result.selects[0].selectedText, "Static (Spread)");
  assert.equal(result.bodyText, "BTC (asset X)\nETH (asset Y)\nHedge ratio 1.25");
});

test("only the projection runs, without a browser, network or download API", () => {
  assert.doesNotMatch(helpers + capture, /downloadJson\(|setSelect\(|fetch\(|localStorage|sessionStorage|document\.cookie/);
  assert.equal(project([]).inputs.length, 0);
});
