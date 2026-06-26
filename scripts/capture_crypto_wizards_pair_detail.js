/*
Paste this into the browser console while on:

  https://cryptowizards.net/wizards/zscore/pair/<id>?origin=scanner

Then click the pair page refresh/recalculate control. After the page finishes
loading, check whether useful payloads were captured:

  await __CW_CAPTURE_STATUS__()

If the status shows useful field hits or a clear next capture focus, run:

  await __CW_DOWNLOAD_CAPTURE__()

It downloads a JSON capture that can be moved to:

  data/raw/pair_details/

Then run:

  PYTHONPATH=src python3 -m quant_platform.cli ingest-pair-details
  PYTHONPATH=src python3 -m quant_platform.cli run-pair-detail-experiments

The script is intentionally local-only: it does not send data anywhere.
*/
(() => {
  const capture = {
    captured_at: new Date().toISOString(),
    url: location.href,
    title: document.title,
    fetches: [],
    xhrs: [],
    worker_messages: [],
    worker_scripts: [],
    wasm_extracts: [],
    resources: [],
    storage: [],
    indexeddb: [],
    scripts: [],
    capture_summary: {},
    page_text: document.body ? document.body.innerText : "",
  };
  let workerId = 0;

  const MAX_TEXT_LENGTH = 250000;
  const MAX_STORAGE_VALUE_LENGTH = 5000;
  const RESEARCH_KEY_PATTERN = /zscore|pair|worker|wasm|api|backtest|spread|ecm|copula|hurst|half|cointegration|view|wizard/i;
  const SENSITIVE_KEY_PATTERN = /token|auth|secret|password|private|key|session|jwt|bearer|cookie/i;
  const FIELD_ALIASES = {
    spread: ["spread", "spreads"],
    zscore: ["zscore", "zscores", "zscore_last", "zscore_roll", "zscore_rolls"],
    ecm_x: ["ecm_x", "ecm_xs"],
    ecm_y: ["ecm_y", "ecm_ys"],
    ecm_strength: ["ecm_strength", "ecm_strengths"],
    price_x: [
      "price_x",
      "prices_x",
      "x_price",
      "x_prices",
      "symbol_1_price",
      "symbol_1_prices",
      "symbol1_price",
      "symbol1_prices",
      "symbol_1_closes",
      "symbol1_closes",
      "series_1_closes",
      "series1_closes",
      "close_x",
      "closes_x",
    ],
    price_y: [
      "price_y",
      "prices_y",
      "y_price",
      "y_prices",
      "symbol_2_price",
      "symbol_2_prices",
      "symbol2_price",
      "symbol2_prices",
      "symbol_2_closes",
      "symbol2_closes",
      "series_2_closes",
      "series2_closes",
      "close_y",
      "closes_y",
    ],
  };
  const BASELINE_REQUIRED_FIELDS = ["spread", "zscore"];
  const ECM_REQUIRED_FIELDS = ["ecm_x", "ecm_y", "ecm_strength"];
  const TWO_LEG_REQUIRED_FIELDS = ["price_x", "price_y"];
  const FIELD_ALIAS_TO_CANONICAL = Object.fromEntries(
    Object.entries(FIELD_ALIASES).flatMap(([canonical, aliases]) =>
      aliases.map((alias) => [alias.replace(/[^a-z0-9]/gi, "").toLowerCase(), canonical])
    )
  );

  const safeClone = (value) => {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (_error) {
      return String(value);
    }
  };

  const maybeResponseText = async (response) => {
    try {
      const text = await response.clone().text();
      return text.length > MAX_TEXT_LENGTH ? `${text.slice(0, MAX_TEXT_LENGTH)}...[truncated]` : text;
    } catch (_error) {
      return null;
    }
  };

  const maybeParseJsonText = (text) => {
    if (!text || typeof text !== "string") return null;
    try {
      return JSON.parse(text);
    } catch (_error) {
      return null;
    }
  };

  const canonicalFieldName = (key) =>
    FIELD_ALIAS_TO_CANONICAL[String(key || "").replace(/[^a-z0-9]/gi, "").toLowerCase()] || null;

  const addFieldHit = (hits, field, path) => {
    if (!field || hits[field]) return;
    hits[field] = path;
  };

  const scanResearchFieldHits = (value, path, hits, seen, depth = 0) => {
    if (value === null || value === undefined || depth > 12) return;
    if (typeof value === "string") {
      const parsed = maybeParseJsonText(value);
      if (parsed && typeof parsed === "object") {
        scanResearchFieldHits(parsed, path, hits, seen, depth + 1);
      }
      return;
    }
    if (typeof value !== "object") return;
    if (seen.has(value)) return;
    seen.add(value);
    if (Array.isArray(value)) {
      value.slice(0, 50).forEach((item, index) => scanResearchFieldHits(item, `${path}[${index}]`, hits, seen, depth + 1));
      return;
    }
    Object.entries(value).forEach(([key, nested]) => {
      if (key === "capture_summary") return;
      const nestedPath = `${path}.${key}`;
      addFieldHit(hits, canonicalFieldName(key), nestedPath);
      scanResearchFieldHits(nested, nestedPath, hits, seen, depth + 1);
    });
  };

  const missingFields = (required, hits) => required.filter((field) => !hits[field]);

  const nextCaptureFocus = (hits) => {
    const missingBaseline = missingFields(BASELINE_REQUIRED_FIELDS, hits);
    const missingEcm = missingFields(ECM_REQUIRED_FIELDS, hits);
    const missingTwoLeg = missingFields(TWO_LEG_REQUIRED_FIELDS, hits);
    if (missingBaseline.length) return `capture_baseline_history:${missingBaseline.join(";")}`;
    if (missingEcm.length) return `capture_ecm_history:${missingEcm.join(";")}`;
    if (missingTwoLeg.length) return `capture_two_leg_prices:${missingTwoLeg.join(";")}`;
    return "capture_ready_for_python_preflight";
  };

  const captureOperatorHint = (summary, quality) => {
    const sourceCount =
      (summary.fetches || 0) +
      (summary.xhrs || 0) +
      (summary.worker_messages || 0) +
      (summary.storage || 0) +
      (summary.indexeddb || 0) +
      (summary.scripts || 0) +
      (summary.resources || 0);
    const focus = quality.next_capture_focus || "unknown";
    if (!sourceCount) {
      return "click_refresh_or_recalculate_before_download";
    }
    if (focus.startsWith("capture_baseline_history")) {
      return "refresh_pair_detail_until_spread_and_zscore_arrays_are_captured";
    }
    if (focus.startsWith("capture_ecm_history")) {
      return "select_ecm_or_dependency_view_then_refresh";
    }
    if (focus.startsWith("capture_two_leg_prices")) {
      return "capture_payload_with_leg_price_history_or_export_network_har";
    }
    return "download_capture_and_run_python_preflight";
  };

  const captureFieldQuality = () => {
    const hits = {};
    scanResearchFieldHits(capture, "$", hits, new WeakSet());
    return {
      required_field_hits: hits,
      missing_baseline_fields: missingFields(BASELINE_REQUIRED_FIELDS, hits),
      missing_ecm_fields: missingFields(ECM_REQUIRED_FIELDS, hits),
      missing_two_leg_fields: missingFields(TWO_LEG_REQUIRED_FIELDS, hits),
      next_capture_focus: nextCaptureFocus(hits),
    };
  };

  const safeStorageValue = (key, value) => {
    if (SENSITIVE_KEY_PATTERN.test(key)) return "[redacted-sensitive-key]";
    if (typeof value !== "string") return safeClone(value);
    return value.length > MAX_STORAGE_VALUE_LENGTH
      ? `${value.slice(0, MAX_STORAGE_VALUE_LENGTH)}...[truncated]`
      : value;
  };

  const selectedValue = (selects, index) => {
    const select = selects[index];
    return select ? select.value : "";
  };

  const inputValue = (inputs, index) => {
    const input = inputs[index];
    return input ? input.value : "";
  };

  const livePairContext = () => {
    const inputs = Array.from(document.querySelectorAll("input"));
    const selects = Array.from(document.querySelectorAll("select"));
    const strategyParts = selectedValue(selects, 1).split("-").map((value) => Number(value));
    const pairIdMatch = location.pathname.match(/pair\/([^/?#]+)/);
    return {
      pair_id: pairIdMatch ? pairIdMatch[1] : null,
      symbol_1: inputValue(inputs, 0),
      symbol_2: inputValue(inputs, 1),
      exchange: "dydx",
      interval: selectedValue(selects, 0) || "daily",
      period: Number(inputValue(inputs, 2)) || null,
      spread_id: strategyParts[0] || null,
      strategy_id: strategyParts[1] || null,
      selected_strategy_value: selectedValue(selects, 1),
    };
  };

  const uniqueIdCandidates = (context) => {
    const { symbol_1: x, symbol_2: y, exchange, interval, period } = context;
    if (!x || !y || !exchange || !interval || !period) return [];
    const intervalAliases = Array.from(new Set([interval, interval === "daily" ? "1d" : interval]));
    const separators = ["-", "_", ":", "|", "::"];
    const orderedParts = (alias) => [
      [x, y, exchange, alias, period],
      [x, y, exchange, period, alias],
      [exchange, x, y, alias, period],
      [exchange, x, y, period, alias],
      [x, y, alias, period],
      [x, y, period, alias],
    ];
    const candidates = [];
    intervalAliases.forEach((alias) => {
      orderedParts(alias).forEach((parts) => {
        separators.forEach((sep) => candidates.push(parts.join(sep)));
      });
    });
    return Array.from(new Set(candidates));
  };

  const tryWasmExtraction = async () => {
    const context = livePairContext();
    const attempts = {
      at: new Date().toISOString(),
      context,
      module_url: "/_build/assets/zscore_library-IlN0_w2C.js",
      candidates: [],
      errors: [],
    };
    try {
      const module = await import("/_build/assets/zscore_library-IlN0_w2C.js");
      if (typeof module.w !== "function") {
        attempts.errors.push("module_export_w_missing");
        capture.wasm_extracts.push(attempts);
        return attempts;
      }
      for (const uniqueId of uniqueIdCandidates(context)) {
        try {
          const value = module.w(uniqueId);
          attempts.candidates.push({
            unique_id: uniqueId,
            ok: true,
            value: safeClone(value),
          });
        } catch (error) {
          attempts.candidates.push({
            unique_id: uniqueId,
            ok: false,
            error: String(error && error.message ? error.message : error),
          });
        }
      }
    } catch (error) {
      attempts.errors.push(String(error && error.message ? error.message : error));
    }
    capture.wasm_extracts.push(attempts);
    return attempts;
  };

  const captureStorageArea = (label, storage) => {
    const rows = [];
    if (!storage) return rows;
    for (let idx = 0; idx < storage.length; idx += 1) {
      const key = storage.key(idx);
      if (!key || (!RESEARCH_KEY_PATTERN.test(key) && SENSITIVE_KEY_PATTERN.test(key))) continue;
      const value = storage.getItem(key);
      rows.push({
        area: label,
        key,
        sensitive: SENSITIVE_KEY_PATTERN.test(key),
        json: SENSITIVE_KEY_PATTERN.test(key) ? null : maybeParseJsonText(value),
        value: safeStorageValue(key, value),
      });
    }
    return rows;
  };

  const refreshStorage = () => {
    capture.storage = [
      ...captureStorageArea("localStorage", window.localStorage),
      ...captureStorageArea("sessionStorage", window.sessionStorage),
    ];
  };

  const refreshInlineScripts = () => {
    capture.scripts = Array.from(document.scripts || [])
      .map((script, index) => {
        const src = script.src || "";
        const text = script.src ? "" : script.textContent || "";
        if (!RESEARCH_KEY_PATTERN.test(src) && !RESEARCH_KEY_PATTERN.test(text)) return null;
        return {
          index,
          src,
          type: script.type || "",
          json: script.type && script.type.includes("json") ? maybeParseJsonText(text) : null,
          text: text ? text.slice(0, MAX_TEXT_LENGTH) : "",
        };
      })
      .filter(Boolean);
  };

  const readObjectStore = (db, storeName) =>
    new Promise((resolve) => {
      const rows = [];
      try {
        const tx = db.transaction(storeName, "readonly");
        const store = tx.objectStore(storeName);
        const request = store.openCursor();
        request.onsuccess = () => {
          const cursor = request.result;
          if (!cursor || rows.length >= 50) {
            resolve(rows);
            return;
          }
          const key = String(cursor.key);
          const value = cursor.value;
          const serialized = JSON.stringify(safeClone(value));
          if (RESEARCH_KEY_PATTERN.test(key) || RESEARCH_KEY_PATTERN.test(serialized || "")) {
            rows.push({
              key,
              value: serialized && serialized.length > MAX_STORAGE_VALUE_LENGTH
                ? `${serialized.slice(0, MAX_STORAGE_VALUE_LENGTH)}...[truncated]`
                : safeClone(value),
            });
          }
          cursor.continue();
        };
        request.onerror = () => resolve(rows);
      } catch (_error) {
        resolve(rows);
      }
    });

  const refreshIndexedDB = async () => {
    capture.indexeddb = [];
    if (!window.indexedDB || !indexedDB.databases) return;
    let databases = [];
    try {
      databases = await indexedDB.databases();
    } catch (_error) {
      return;
    }
    for (const info of databases) {
      const name = info && info.name;
      if (!name || (!RESEARCH_KEY_PATTERN.test(name) && SENSITIVE_KEY_PATTERN.test(name))) continue;
      const dbRows = {
        name,
        version: info.version,
        stores: [],
      };
      await new Promise((resolve) => {
        const open = indexedDB.open(name);
        open.onerror = () => resolve();
        open.onsuccess = async () => {
          const db = open.result;
          try {
            for (const storeName of Array.from(db.objectStoreNames || [])) {
              if (!RESEARCH_KEY_PATTERN.test(storeName) && SENSITIVE_KEY_PATTERN.test(storeName)) continue;
              const rows = await readObjectStore(db, storeName);
              dbRows.stores.push({ name: storeName, rows });
            }
          } finally {
            db.close();
            resolve();
          }
        };
      });
      capture.indexeddb.push(dbRows);
    }
  };

  if (!window.__CW_ORIGINAL_FETCH__) {
    window.__CW_ORIGINAL_FETCH__ = window.fetch;
    window.fetch = async (...args) => {
      const startedAt = new Date().toISOString();
      const url = String(args[0] && args[0].url ? args[0].url : args[0]);
      const response = await window.__CW_ORIGINAL_FETCH__(...args);
      const text = await maybeResponseText(response);
      capture.fetches.push({
        started_at: startedAt,
        url,
        status: response.status,
        content_type: response.headers.get("content-type"),
        json: maybeParseJsonText(text),
        text,
      });
      return response;
    };
  }

  if (window.XMLHttpRequest && !window.__CW_XHR_PATCHED__) {
    window.__CW_XHR_PATCHED__ = true;
    const originalOpen = XMLHttpRequest.prototype.open;
    const originalSend = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function patchedOpen(method, url, ...rest) {
      this.__cw_method = method;
      this.__cw_url = String(url);
      return originalOpen.call(this, method, url, ...rest);
    };
    XMLHttpRequest.prototype.send = function patchedSend(body) {
      const startedAt = new Date().toISOString();
      this.addEventListener("loadend", () => {
        let responseText = null;
        try {
          responseText =
            typeof this.responseText === "string" ? this.responseText.slice(0, MAX_TEXT_LENGTH) : null;
        } catch (_error) {
          responseText = null;
        }
        capture.xhrs.push({
          started_at: startedAt,
          url: this.__cw_url,
          method: this.__cw_method,
          status: this.status,
          content_type: this.getResponseHeader("content-type"),
          json: maybeParseJsonText(responseText),
          text: responseText,
          body: typeof body === "string" ? body.slice(0, MAX_TEXT_LENGTH) : safeClone(body),
        });
      });
      return originalSend.call(this, body);
    };
  }

  if (window.Worker && !window.__CW_WORKER_PATCHED__) {
    window.__CW_WORKER_PATCHED__ = true;
    const OriginalWorker = window.Worker;
    const workerMeta = new WeakMap();
    let originalAddEventListener = null;

    const workerInfo = (worker) =>
      workerMeta.get(worker) || {
        worker_id: "unknown",
        script_url: "",
      };

    const captureWorkerMessage = (worker, direction, message) => {
      const meta = workerInfo(worker);
      capture.worker_messages.push({
        direction,
        at: new Date().toISOString(),
        worker_id: meta.worker_id,
        script_url: meta.script_url,
        message: safeClone(message),
      });
    };

    const attachPassiveWorkerCapture = (worker) => {
      if (!worker || worker.__cw_passive_capture_attached) return;
      try {
        Object.defineProperty(worker, "__cw_passive_capture_attached", {
          value: true,
          configurable: false,
          enumerable: false,
        });
        const addListener = originalAddEventListener || worker.addEventListener;
        addListener.call(worker, "message", (event) => {
          captureWorkerMessage(worker, "from_worker", event.data);
        });
        addListener.call(worker, "messageerror", (event) => {
          captureWorkerMessage(worker, "from_worker_messageerror", event.data);
        });
      } catch (_error) {
        // Some browser Worker implementations may prevent property definition.
      }
    };

    window.Worker = function PatchedWorker(scriptURL, options) {
      const id = `worker_${workerId += 1}`;
      capture.worker_scripts.push({
        at: new Date().toISOString(),
        worker_id: id,
        script_url: String(scriptURL),
        options: safeClone(options || {}),
      });
      const worker = new OriginalWorker(scriptURL, options);
      workerMeta.set(worker, {
        worker_id: id,
        script_url: String(scriptURL),
      });
      attachPassiveWorkerCapture(worker);
      return worker;
    };
    window.Worker.prototype = OriginalWorker.prototype;
    window.Worker.__CW_ORIGINAL_WORKER__ = OriginalWorker;

    const originalPostMessage = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function patchedPostMessage(message, transfer) {
      captureWorkerMessage(this, "to_worker", message);
      return originalPostMessage.call(this, message, transfer);
    };

    originalAddEventListener = Worker.prototype.addEventListener;
    Worker.prototype.addEventListener = function patchedAddEventListener(type, listener, options) {
      if (type !== "message" || typeof listener !== "function") {
        return originalAddEventListener.call(this, type, listener, options);
      }
      const wrapped = function wrappedWorkerMessage(event) {
        captureWorkerMessage(this, "from_worker_listener", event.data);
        return listener.call(this, event);
      };
      return originalAddEventListener.call(this, type, wrapped, options);
    };

    const originalOnMessage = Object.getOwnPropertyDescriptor(Worker.prototype, "onmessage");
    if (!originalOnMessage || originalOnMessage.configurable !== false) {
      Object.defineProperty(Worker.prototype, "onmessage", {
        configurable: true,
        enumerable: true,
        get() {
          return this.__cw_onmessage || null;
        },
        set(listener) {
          if (typeof listener !== "function") {
            this.__cw_onmessage = listener;
            return;
          }
          this.__cw_onmessage = function wrappedOnMessage(event) {
            captureWorkerMessage(this, "from_worker_onmessage", event.data);
            return listener.call(this, event);
          };
          if (originalOnMessage && originalOnMessage.set) {
            originalOnMessage.set.call(this, this.__cw_onmessage);
          } else {
            this.addEventListener("message", this.__cw_onmessage);
          }
        },
      });
    }
  }

  window.__CW_CAPTURE__ = capture;
  window.__CW_CAPTURE_SUMMARY__ = () => ({
    fetches: capture.fetches.length,
    xhrs: capture.xhrs.length,
    worker_messages: capture.worker_messages.length,
    worker_scripts: capture.worker_scripts.length,
    wasm_extracts: capture.wasm_extracts.length,
    resources: capture.resources.length,
    storage: capture.storage.length,
    indexeddb: capture.indexeddb.length,
    scripts: capture.scripts.length,
    page_text_length: capture.page_text ? capture.page_text.length : 0,
    has_network_payloads: capture.fetches.length + capture.xhrs.length > 0,
    has_worker_payloads: capture.worker_messages.length > 0,
    has_wasm_extracts: capture.wasm_extracts.length > 0,
    has_storage_payloads: capture.storage.length + capture.indexeddb.length > 0,
    field_quality: captureFieldQuality(),
  });
  window.__CW_REFRESH_CAPTURE__ = async () => {
    capture.captured_at = new Date().toISOString();
    capture.url = location.href;
    capture.title = document.title;
    capture.page_text = document.body ? document.body.innerText : "";
    refreshStorage();
    refreshInlineScripts();
    await tryWasmExtraction();
    await refreshIndexedDB();
    capture.resources = performance
      .getEntriesByType("resource")
      .map((entry) => ({
        name: entry.name,
        initiator_type: entry.initiatorType,
        duration: entry.duration,
        transfer_size: entry.transferSize,
      }))
      .filter((entry) => RESEARCH_KEY_PATTERN.test(entry.name));
    capture.capture_summary = window.__CW_CAPTURE_SUMMARY__();
    return capture;
  };
  window.__CW_CAPTURE_STATUS__ = async () => {
    await window.__CW_REFRESH_CAPTURE__();
    const summary = capture.capture_summary;
    const quality = summary.field_quality || {};
    const status = {
      fetches: summary.fetches,
      xhrs: summary.xhrs,
      worker_messages: summary.worker_messages,
      wasm_extracts: summary.wasm_extracts,
      storage: summary.storage,
      indexeddb: summary.indexeddb,
      scripts: summary.scripts,
      resources: summary.resources,
      required_field_hits: Object.keys(quality.required_field_hits || {}).join(";") || "none",
      missing_baseline_fields: (quality.missing_baseline_fields || []).join(";") || "none",
      missing_ecm_fields: (quality.missing_ecm_fields || []).join(";") || "none",
      missing_two_leg_fields: (quality.missing_two_leg_fields || []).join(";") || "none",
      next_capture_focus: quality.next_capture_focus || "unknown",
      capture_operator_hint: captureOperatorHint(summary, quality),
    };
    console.table([status]);
    return status;
  };
  window.__CW_CAPTURE_RUNBOOK__ = async () => {
    const status = await window.__CW_CAPTURE_STATUS__();
    return [
      `next_capture_focus=${status.next_capture_focus}`,
      `capture_operator_hint=${status.capture_operator_hint}`,
      "1. Run: await __CW_CAPTURE_STATUS__() to attempt direct WASM extraction.",
      "2. Click the Crypto Wizards pair refresh/recalculate control.",
      "3. Run: await __CW_CAPTURE_STATUS__() again.",
      "4. If required fields are hit or wasm_extracts/network/worker payloads increased, run: await __CW_DOWNLOAD_CAPTURE__()",
      "5. Move the JSON into data/raw/pair_details/ and run capture-preflight.",
    ].join("\n");
  };
  window.__CW_DOWNLOAD_CAPTURE__ = async () => {
    const data = await window.__CW_REFRESH_CAPTURE__();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const pairMatch = location.pathname.match(/pair\/([^/?#]+)/);
    const pairId = pairMatch ? pairMatch[1] : "unknown";
    link.href = url;
    link.download = `crypto_wizards_pair_${pairId}_capture.json`;
    link.click();
    URL.revokeObjectURL(url);
    return data;
  };

  console.log(
    "Crypto Wizards capture installed. Run __CW_CAPTURE_STATUS__(), click refresh/recalculate, run status again, then run __CW_DOWNLOAD_CAPTURE__()."
  );
})();
