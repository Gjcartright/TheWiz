/*
Paste this into the browser console while on:

  https://cryptowizards.net/wizards/zscore/scanner

Then run:

  await __CW_DOWNLOAD_SCANNER_ROWS__({
    requestedUrl: "https://cryptowizards.net/wizards/zscore/scanner"
  })

Move the downloaded JSON into:

  data/raw/crypto_wizards_scanner/

Then run:

  PYTHONPATH=src python3 -m quant_platform.cli ingest-crypto-wizards-scanner

The helper captures visible scanner-table rows only. It preserves raw cell text
so the Python normalizer can retain every visible column even if the UI shifts.
*/
(() => {
  const text = (node) => (node ? node.innerText.replace(/\s+/g, " ").trim() : "");
  const lower = (value) => String(value || "").trim().toLowerCase();

  const memberNavigationTargets = () => Array.from(document.querySelectorAll("a[href]"))
    .map((anchor) => {
      try {
        const url = new URL(anchor.href, location.href);
        return url.protocol === "https:" && /^(www\.)?cryptowizards\.net$/i.test(url.hostname)
          && url.pathname.startsWith("/wizards/")
          ? url.href
          : "";
      } catch (_) {
        return "";
      }
    })
    .filter(Boolean)
    .filter((value, index, values) => values.indexOf(value) === index);

  const browserAuthObservation = (requestedUrl) => {
    const selects = Array.from(document.querySelectorAll("select")).filter(visible);
    const bodyText = document.body?.innerText || "";
    const resultSurface = Boolean(
      document.querySelector("table, [role='grid'], [role='table']")
      || (/\b\d+\s+results\b/i.test(bodyText) && /[A-Z0-9]+-USD/.test(bodyText)),
    );
    const formsText = lower(Array.from(document.querySelectorAll("form"))
      .map((form) => form.innerText || "")
      .join(" "));
    const markers = [];
    if (selects.length >= 6) markers.push("scanner_filter_controls");
    if (selects.some((select) => Array.from(select.options).some(
      (option) => /spread|copula|zscore/i.test(option.text),
    ))) markers.push("scanner_strategy_control");
    if (selects.some((select) => Array.from(select.options).some(
      (option) => /binance|coinbase|dydx|bybit/i.test(option.text),
    ))) markers.push("scanner_exchange_control");
    if (resultSurface) markers.push("scanner_results_surface");
    return {
      schema_version: "wizard_browser_auth_observation.v1",
      captured_at: new Date().toISOString(),
      requested_url: requestedUrl || location.href,
      requested_url_source: requestedUrl ? "capture_argument" : "current_location_fallback",
      final_url: location.href,
      route_kind: "scanner",
      member_navigation_targets: memberNavigationTargets(),
      protected_content_markers: markers,
      sign_in_form_present: Boolean(document.querySelector('input[type="password"]'))
        || (/\b(sign in|log in)\b/.test(formsText)
          && Boolean(document.querySelector('input[type="email"], input[name*="email" i]'))),
      verification_form_present: Boolean(document.querySelector('input[autocomplete="one-time-code"]'))
        || /\b(verification code|security code|one-time code)\b/.test(formsText),
      public_marketing_shell_present: !location.pathname.startsWith("/wizards/"),
      browser_storage_accessed: false,
      no_credentials_or_browser_storage_captured: true,
    };
  };

  const requireAuthenticatedScannerRoute = (observation) => {
    const required = [
      "scanner_filter_controls",
      "scanner_strategy_control",
      "scanner_exchange_control",
      "scanner_results_surface",
    ];
    const accountPresent = observation.member_navigation_targets.some((value) => {
      try {
        return new URL(value).pathname.replace(/\/$/, "") === "/wizards/account";
      } catch (_) {
        return false;
      }
    });
    const checks = {
      explicit_requested_url: observation.requested_url_source === "capture_argument",
      protected_scanner_route: location.pathname.replace(/\/$/, "") === "/wizards/zscore/scanner",
      member_navigation: observation.member_navigation_targets.length >= 2,
      account_navigation: accountPresent,
      route_controls: required.every((value) => observation.protected_content_markers.includes(value)),
      no_sign_in_form: !observation.sign_in_form_present,
      no_verification_form: !observation.verification_form_present,
      no_public_shell: !observation.public_marketing_shell_present,
    };
    if (!Object.values(checks).every(Boolean)) {
      console.table(checks);
      throw new Error(`Authenticated scanner-route proof failed: ${JSON.stringify(checks)}`);
    }
  };

  const selectedText = (select) => {
    if (!select) return "";
    const option = select.options && select.selectedIndex >= 0 ? select.options[select.selectedIndex] : null;
    return option ? option.textContent.trim() : select.value;
  };

  const visible = (element) => {
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
  };

  const captureFilters = () => {
    const selects = Array.from(document.querySelectorAll("select")).filter(visible);
    const inputs = Array.from(document.querySelectorAll("input")).filter(visible);
    return {
      priority: selectedText(selects[0]),
      count: selectedText(selects[1]),
      correlation: selectedText(selects[2]),
      hurst: selectedText(selects[3]),
      half_life: selectedText(selects[4]),
      copula: selectedText(selects[5]),
      strategy: selectedText(selects[6]),
      symbol: inputs.map((input) => input.value).filter(Boolean).join(";"),
      exchange: selectedText(selects[7]) || selectedText(selects[selects.length - 1]),
    };
  };

  const rowCells = (row) => {
    const cells = Array.from(row.querySelectorAll(":scope > td, :scope > th"));
    if (cells.length) return cells.map(text);
    return Array.from(row.children).map(text).filter(Boolean);
  };

  const badgeColor = (svg) => {
    if (!svg) return "unknown";
    const painted = svg.querySelector("path, circle, rect") || svg;
    const token = `${svg.getAttribute("class") || ""} ${painted.getAttribute("class") || ""}`.toLowerCase();
    const fill = getComputedStyle(painted).fill || getComputedStyle(svg).fill || "";
    if (/success|green|emerald/.test(token) || /rgb\(0,\s*139,\s*15\)/.test(fill)) return "green";
    if (/warning|orange|amber/.test(token) || /rgb\(255,\s*144,\s*0\)/.test(fill)) return "orange";
    if (/natural|gray|grey/.test(token) || /rgb\(93,\s*115,\s*134\)/.test(fill)) return "gray";
    return "unknown";
  };

  const stationarityBadges = (row) => {
    const cell = Array.from(row.querySelectorAll(":scope > td, :scope > th"))[7];
    if (!cell) return { badges: {}, details: {} };
    const result = { badges: {}, details: {} };
    for (const [key, label] of [["jn", "Jn"], ["eg", "EG"]]) {
      const labelNode = Array.from(cell.querySelectorAll("div, span, p"))
        .find((node) => text(node) === label);
      const container = labelNode?.closest(".flex-center-col") || labelNode?.parentElement;
      const svg = container?.querySelector("svg") || null;
      const color = badgeColor(svg);
      result.badges[key] = color;
      result.details[key] = {
        color,
        svg_class: svg?.getAttribute("class") || "",
        computed_fill: svg ? (getComputedStyle(svg.querySelector("path, circle, rect") || svg).fill || "") : "",
      };
    }
    return result;
  };

  const rowObject = (row, index) => {
    const cells = rowCells(row);
    const stationarity = stationarityBadges(row);
    return {
      row_index: index,
      cells,
      raw_pair_cell: cells[0] || "",
      raw_volume_cell: cells[1] || "",
      raw_spread_cell: cells[2] || "",
      raw_updated_cell: cells[3] || "",
      raw_strategy_cell: cells[4] || "",
      raw_zscore_cell: cells[5] || "",
      raw_dependency_cell: cells[6] || "",
      raw_stationarity_cell: cells[7] || "",
      raw_risk_cell: cells[8] || "",
      raw_reward_cell: cells[9] || "",
      stationarity_badges: stationarity.badges,
      stationarity_badge_details: stationarity.details,
    };
  };

  const captureRows = () => {
    const tables = Array.from(document.querySelectorAll("table"));
    const tableRows = tables.flatMap((table) => Array.from(table.querySelectorAll("tbody tr, tr")));
    const candidateRows = tableRows.length
      ? tableRows
      : Array.from(document.querySelectorAll("[role='row'], .row, [class*='row']"));
    return candidateRows
      .filter(visible)
      .map(rowObject)
      .filter((row) => row.cells.length >= 8 && /[A-Z0-9]+-USD/.test(row.raw_pair_cell))
      .map((row, index) => ({ ...row, row_index: index }));
  };

  const capture = ({ requestedUrl = "" } = {}) => {
    const authObservation = browserAuthObservation(requestedUrl);
    requireAuthenticatedScannerRoute(authObservation);
    return {
    schema_version: "crypto_wizards_scanner_browser_capture.v2",
    captured_at: new Date().toISOString(),
    url: location.href,
    title: document.title,
    browser_auth_observation: authObservation,
    no_credentials_or_browser_storage_captured: true,
    scanner_filters: captureFilters(),
    scanner_rows: captureRows(),
  };
  };

  window.__CW_CAPTURE_SCANNER_ROWS__ = capture;
  window.__CW_DOWNLOAD_SCANNER_ROWS__ = async (options = {}) => {
    const payload = capture(options);
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    link.href = URL.createObjectURL(blob);
    link.download = `crypto_wizards_scanner_rows_${timestamp}.json`;
    document.body.appendChild(link);
    link.click();
    URL.revokeObjectURL(link.href);
    link.remove();
    return {
      rows: payload.scanner_rows.length,
      filters: payload.scanner_filters,
      download: link.download,
    };
  };

  console.log("Crypto Wizards scanner capture ready. Pass the requested scanner URL explicitly.");
})();
