/*
Paste this script into the browser console on an authenticated Crypto Wizards
pair page. It captures one explicitly verified orientation across every mode
currently exposed by the pair page.

For reverse orientation, create a new Custom Analysis tab with asset Y entered
as X and asset X entered as Y. The capture refuses to run until both the inputs
and the rendered "asset X" / "asset Y" labels prove that order.

Example:

  await __CW_CAPTURE_PAIR_UI_BUNDLE__({
    orientation: "reverse",
    scannerContext: {
      exchange: "binance",
      interval: "daily",
      asset_x_raw: "BTCUSDT",
      asset_y_raw: "MOVEUSDT",
    },
    scannerEvidencePath: "data/raw/.../binance_daily_all_rows.json",
  });
*/
(() => {
  const MODES = [
    { value: "3-1", label: "Static (Spread)" },
    { value: "3-2", label: "Static (ZScoreR)" },
    { value: "1-1", label: "Dyn (Spread)" },
    { value: "1-2", label: "Dyn (ZScoreR)" },
    { value: "2-1", label: "OU (Spread)" },
    { value: "2-2", label: "OU (ZScoreR)" },
    { value: "1-3", label: "Copula" },
  ];
  const MODE_SELECTOR = 'select:has(option[value="3-1"])';
  const TIMEFRAME_SELECTOR = 'select:has(option[value="daily"])';
  const CONDITIONAL_SELECTOR = 'select:has(option[value="prices_1"])';
  const DEPENDENCY_SELECTOR = 'select:has(option[value="ecm_strength"])';
  const BACKTEST_SELECTOR = 'select:has(option[value="underwater"])';

  const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
  const upper = (value) => String(value || "").trim().toUpperCase();
  const lower = (value) => String(value || "").trim().toLowerCase();

  const setSelect = async (selector, value, waitMilliseconds) => {
    const select = document.querySelector(selector);
    if (!select) throw new Error(`Missing selector: ${selector}`);
    select.value = value;
    select.dispatchEvent(new Event("change", { bubbles: true }));
    await sleep(waitMilliseconds);
  };

  const renderedAssetOrder = () => {
    const assets = {};
    for (const line of (document.body?.innerText || "").split("\n")) {
      const match = line.trim().match(/^(.+?)\s+\(asset\s+([XY])\)$/i);
      if (match) assets[match[2].toUpperCase()] = upper(match[1]);
    }
    return [assets.X || "", assets.Y || ""];
  };

  const visibleExchange = () => {
    const lines = (document.body?.innerText || "")
      .split("\n")
      .map((line) => lower(line))
      .filter(Boolean);
    const periodIndex = lines.findIndex((line) => line.includes("periods analyzed"));
    return periodIndex > 0 ? lines[periodIndex - 1] : "";
  };

  const orientationStatus = ({ scannerContext, orientation }) => {
    const inputs = Array.from(document.querySelectorAll("input"));
    const inputOrder = [upper(inputs[0]?.value), upper(inputs[1]?.value)];
    const originalOrder = [
      upper(scannerContext.asset_x_raw),
      upper(scannerContext.asset_y_raw),
    ];
    const expectedOrder = orientation === "reverse"
      ? [originalOrder[1], originalOrder[0]]
      : originalOrder;
    const renderedOrder = renderedAssetOrder();
    const timeframe = lower(document.querySelector(TIMEFRAME_SELECTOR)?.value);
    const exchange = visibleExchange();
    const checks = {
      orientation_supported: ["original", "reverse"].includes(orientation),
      scanner_identity_complete: originalOrder.every(Boolean),
      input_order_matches: inputOrder.join("|") === expectedOrder.join("|"),
      rendered_order_matches: renderedOrder.join("|") === expectedOrder.join("|"),
      input_and_rendered_match: inputOrder.join("|") === renderedOrder.join("|"),
      timeframe_matches: timeframe === lower(scannerContext.interval),
      exchange_matches: exchange === lower(scannerContext.exchange),
    };
    return {
      valid: Object.values(checks).every(Boolean),
      checks,
      orientation,
      expectedOrder,
      inputOrder,
      renderedOrder,
      timeframe,
      exchange,
      url: location.href,
    };
  };

  const captureState = () => {
    const selects = Array.from(document.querySelectorAll("select")).map((element, index) => ({
      index,
      value: element.value,
      selectedText: element.options[element.selectedIndex]?.text || "",
      options: Array.from(element.options).map((option) => ({
        value: option.value,
        text: option.text,
        selected: option.selected,
      })),
    }));
    const inputs = Array.from(document.querySelectorAll("input")).map((element, index) => ({
      index,
      type: element.type,
      value: element.value,
      name: element.name || "",
      min: element.min || "",
      max: element.max || "",
      step: element.step || "",
      context: (element.parentElement?.innerText || "").trim().slice(0, 240),
    }));
    const stationarity = Array.from(document.querySelectorAll("div"))
      .filter((element) => ["coint Jn", "coint EG"].includes((element.innerText || "").trim())
        && element.children.length === 0)
      .map((element) => {
        const container = element.closest(".flex-center-col") || element.parentElement;
        const svg = container?.querySelector("svg");
        return {
          label: (element.innerText || "").trim(),
          svgClass: svg?.getAttribute("class") || "",
          containerClass: container?.getAttribute("class") || "",
        };
      });
    const svgs = Array.from(document.querySelectorAll("svg")).map((svg, index) => ({
      index,
      viewBox: svg.getAttribute("viewBox") || "",
      className: svg.getAttribute("class") || "",
      ariaLabel: svg.getAttribute("aria-label") || "",
      paths: Array.from(svg.querySelectorAll("path")).map((path) => path.getAttribute("d") || ""),
    }));
    return {
      capturedAt: new Date().toISOString(),
      url: location.href,
      title: document.title,
      bodyText: (document.body?.innerText || "").trim(),
      selects,
      inputs,
      stationarity,
      svgs,
      buttons: Array.from(document.querySelectorAll("button")).map((button, index) => ({
        index,
        text: (button.innerText || "").trim(),
        title: button.title || "",
        ariaLabel: button.getAttribute("aria-label") || "",
      })),
    };
  };

  const chartOptions = (selector) => {
    const select = document.querySelector(selector);
    return select
      ? Array.from(select.options).map((option) => ({ value: option.value, text: option.text }))
      : [];
  };

  const captureChartFamily = async (selector, waitMilliseconds) => {
    const captures = [];
    for (const option of chartOptions(selector)) {
      await setSelect(selector, option.value, waitMilliseconds);
      const state = captureState();
      captures.push({
        chart_value: option.value,
        chart_label: option.text,
        captured_at: state.capturedAt,
        bodyText: state.bodyText,
        svgs: state.svgs,
      });
    }
    return captures;
  };

  const safeFilename = (value) => String(value || "")
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");

  const downloadJson = (payload, filename) => {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  };

  window.__CW_ORIENTATION_STATUS__ = (options) => orientationStatus(options);
  window.__CW_CAPTURE_PAIR_UI_BUNDLE__ = async ({
    scannerContext,
    orientation = "original",
    scannerEvidencePath = "",
    download = true,
    filename = "",
  }) => {
    orientation = lower(orientation);
    await setSelect(TIMEFRAME_SELECTOR, lower(scannerContext.interval), 1200);
    const initialProof = orientationStatus({ scannerContext, orientation });
    if (!initialProof.valid) {
      console.table(initialProof.checks);
      throw new Error(`Pair orientation verification failed: ${JSON.stringify(initialProof)}`);
    }

    const timestamp = new Date().toISOString();
    const bundle = {
      schema_version: "wizard_pair_detail_ui_bundle.v1",
      capture_method: "authenticated_browser_ui",
      source_authority: "crypto_wizards_dashboard",
      route_identity_policy: "session_local_route_evidence_only",
      canonical_identity_fields: [
        "capture_run_id",
        "wizard_exchange",
        "timeframe",
        "asset_x",
        "asset_y",
      ],
      capture_run_id: `wizard_pair_${orientation}_${timestamp.replace(/[-:.]/g, "")}`,
      captured_at: timestamp,
      no_credentials_or_browser_storage_captured: true,
      page_route: location.href,
      scanner_evidence_path: scannerEvidencePath,
      scanner_context: scannerContext,
      page_orientation_context: {
        asset_x_raw: initialProof.renderedOrder[0],
        asset_y_raw: initialProof.renderedOrder[1],
        verification: "inputs_and_rendered_asset_labels",
      },
      orientation_capture_method: orientation === "reverse"
        ? "new_custom_analysis_pair_tab_ordered_assets"
        : "scanner_pair_tab_ordered_assets",
      modes_expected: MODES.map((mode) => mode.label),
      modes_not_available_on_pair_page: ["OU (Optimal)"],
      orientations_captured: [orientation],
      orientation_blocker: "",
      mode_captures: [],
      conditional_chart_captures: [],
      dependency_chart_captures: [],
      backtest_chart_captures: [],
    };

    for (const mode of MODES) {
      await setSelect(MODE_SELECTOR, mode.value, 750);
      const proof = orientationStatus({ scannerContext, orientation });
      if (!proof.valid) throw new Error(`Orientation drift in ${mode.label}`);
      bundle.mode_captures.push({
        ...captureState(),
        exact_mode: mode.label,
        mode_value: mode.value,
        orientation,
        orientationProof: proof,
      });
    }

    await setSelect(MODE_SELECTOR, "1-3", 600);
    bundle.conditional_chart_captures = await captureChartFamily(CONDITIONAL_SELECTOR, 300);
    bundle.conditional_chart_status = bundle.conditional_chart_captures.length
      ? "AVAILABLE"
      : "NOT_AVAILABLE_ON_PAIR_PAGE";
    bundle.dependency_chart_captures = await captureChartFamily(DEPENDENCY_SELECTOR, 300);
    bundle.dependency_chart_status = bundle.dependency_chart_captures.length
      ? "AVAILABLE"
      : "NOT_AVAILABLE_ON_PAIR_PAGE";

    const backtestOptions = chartOptions(BACKTEST_SELECTOR);
    for (const mode of MODES) {
      await setSelect(MODE_SELECTOR, mode.value, 600);
      for (const option of backtestOptions) {
        await setSelect(BACKTEST_SELECTOR, option.value, 250);
        bundle.backtest_chart_captures.push({
          exact_mode: mode.label,
          chart_value: option.value,
          chart_label: option.text,
          captured_at: new Date().toISOString(),
          svgs: captureState().svgs,
        });
      }
    }
    bundle.backtest_chart_status = backtestOptions.length
      ? "AVAILABLE"
      : "NOT_AVAILABLE_ON_PAIR_PAGE";
    bundle.completed_at = new Date().toISOString();
    bundle.capture_status = `COMPLETE_${orientation.toUpperCase()}_ORIENTATION`;
    bundle.mode_coverage = {
      expected: MODES.length,
      captured: bundle.mode_captures.length,
      not_available_on_pair_page: 1,
    };
    bundle.chart_coverage = {
      conditional_expected: chartOptions(CONDITIONAL_SELECTOR).length,
      conditional_captured: bundle.conditional_chart_captures.length,
      dependency_expected: chartOptions(DEPENDENCY_SELECTOR).length,
      dependency_captured: bundle.dependency_chart_captures.length,
      backtest_expected: MODES.length * backtestOptions.length,
      backtest_captured: bundle.backtest_chart_captures.length,
    };

    if (download) {
      const outputName = filename || safeFilename([
        lower(scannerContext.exchange),
        lower(scannerContext.interval),
        initialProof.renderedOrder[0],
        initialProof.renderedOrder[1],
        orientation,
        "all_modes.json",
      ].join("_"));
      downloadJson(bundle, outputName);
    }
    return bundle;
  };

  console.log("Crypto Wizards orientation-aware pair capture installed.");
})();
