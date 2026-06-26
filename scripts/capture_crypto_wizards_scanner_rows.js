/*
Paste this into the browser console while on:

  https://cryptowizards.net/wizards/zscore/scanner

Then run:

  await __CW_DOWNLOAD_SCANNER_ROWS__()

Move the downloaded JSON into:

  data/raw/crypto_wizards_scanner/

Then run:

  PYTHONPATH=src python3 -m quant_platform.cli ingest-crypto-wizards-scanner

The helper captures visible scanner-table rows only. It preserves raw cell text
so the Python normalizer can retain every visible column even if the UI shifts.
*/
(() => {
  const text = (node) => (node ? node.innerText.replace(/\s+/g, " ").trim() : "");

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

  const rowObject = (row, index) => {
    const cells = rowCells(row);
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

  const capture = () => ({
    captured_at: new Date().toISOString(),
    url: location.href,
    title: document.title,
    scanner_filters: captureFilters(),
    scanner_rows: captureRows(),
  });

  window.__CW_CAPTURE_SCANNER_ROWS__ = capture;
  window.__CW_DOWNLOAD_SCANNER_ROWS__ = async () => {
    const payload = capture();
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

  console.log("Crypto Wizards scanner capture ready. Run await __CW_DOWNLOAD_SCANNER_ROWS__()");
})();
