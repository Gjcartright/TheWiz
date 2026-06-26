/*
Paste this into the browser console on a Crypto Wizards zscore scanner or pair page.

It finds visible dYdX pairs, fetches 5-minute dYdX candles for both legs from
the public dYdX indexer, and downloads one JSON bundle.

Useful commands after paste:

  await __CW_5MIN_PAIRS__()
  await __CW_DOWNLOAD_5MIN_BUNDLE__({ maxPairs: 10, limit: 100 })

The downloaded bundle can be imported with:

  PYTHONPATH=src python3 -m quant_platform.cli import-dydx-candle-bundle --json-path /path/to/crypto_wizards_5min_pair_bundle.json
*/
(() => {
  const MARKET_PATTERN = /\b[A-Z0-9]+-USD\b/g;
  const INDEXER_BASE = window.__QPA_INDEXER_BASE__ || "https://indexer.dydx.trade";

  const unique = (items) => Array.from(new Set(items));

  const visibleTextLines = () =>
    (document.body ? document.body.innerText : "")
      .split(/\n+/)
      .map((line) => line.trim())
      .filter(Boolean);

  const pairFromLine = (line, index) => {
    const markets = unique(line.match(MARKET_PATTERN) || []);
    if (markets.length < 2) return null;
    const exchange = /\bdydx?\b/i.test(line) ? "dydx" : "unknown";
    return {
      pair_id: String(index + 1),
      pair: `${markets[0]}-${markets[1]}`,
      asset_x: markets[0],
      asset_y: markets[1],
      exchange,
      source_text: line.slice(0, 240),
    };
  };

  const currentPairFromInputs = () => {
    const inputs = Array.from(document.querySelectorAll("input")).map((input) => input.value).filter(Boolean);
    const markets = inputs.filter((value) => MARKET_PATTERN.test(value));
    if (markets.length < 2) return null;
    const pairIdMatch = location.pathname.match(/pair\/([^/?#]+)/);
    return {
      pair_id: pairIdMatch ? pairIdMatch[1] : "current",
      pair: `${markets[0]}-${markets[1]}`,
      asset_x: markets[0],
      asset_y: markets[1],
      exchange: "dydx",
      source_text: "current_pair_inputs",
    };
  };

  const discoverPairs = () => {
    const pairs = [];
    const current = currentPairFromInputs();
    if (current) pairs.push(current);
    visibleTextLines().forEach((line, index) => {
      const pair = pairFromLine(line, index);
      if (pair) pairs.push(pair);
    });
    const seen = new Set();
    return pairs.filter((pair) => {
      const key = `${pair.asset_x}|${pair.asset_y}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return pair.exchange === "dydx" || pair.exchange === "unknown";
    });
  };

  const candleUrl = (market, { toISO, limit }) => {
    const query = new URLSearchParams({
      resolution: "5MINS",
      toISO,
      limit: String(limit),
    });
    return `${INDEXER_BASE}/v4/candles/perpetualMarkets/${encodeURIComponent(market)}?${query.toString()}`;
  };

  const fetchCandles = async (market, options) => {
    const url = candleUrl(market, options);
    const response = await fetch(url);
    const text = await response.text();
    let json = null;
    try {
      json = JSON.parse(text);
    } catch (error) {
      json = { parse_error: String(error), text: text.slice(0, 2000) };
    }
    return {
      market,
      url,
      ok: response.ok,
      status: response.status,
      json,
    };
  };

  window.__CW_5MIN_PAIRS__ = async () => {
    const pairs = discoverPairs();
    console.table(pairs.map(({ pair_id, asset_x, asset_y, exchange }) => ({ pair_id, asset_x, asset_y, exchange })));
    return pairs;
  };

  window.__CW_DOWNLOAD_5MIN_BUNDLE__ = async (config = {}) => {
    const limit = Number(config.limit || 100);
    const maxPairs = Number(config.maxPairs || 20);
    const toISO = config.toISO || new Date().toISOString();
    const discovered = discoverPairs().slice(0, maxPairs);
    const bundle = {
      captured_at: new Date().toISOString(),
      page_url: location.href,
      page_title: document.title,
      indexer_base: INDEXER_BASE,
      resolution: "5MINS",
      limit,
      toISO,
      pairs: [],
    };
    for (const pair of discovered) {
      const assetX = await fetchCandles(pair.asset_x, { toISO, limit });
      const assetY = await fetchCandles(pair.asset_y, { toISO, limit });
      bundle.pairs.push({
        ...pair,
        legs: {
          asset_x: assetX,
          asset_y: assetY,
        },
      });
      console.log(`captured ${pair.asset_x}/${pair.asset_y}`, assetX.status, assetY.status);
    }
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "crypto_wizards_5min_pair_bundle.json";
    link.click();
    URL.revokeObjectURL(url);
    console.log(`Downloaded ${bundle.pairs.length} 5-minute pair bundle(s).`);
    return bundle;
  };

  console.log("Crypto Wizards 5-minute bundle helper installed. Run await __CW_5MIN_PAIRS__(), then await __CW_DOWNLOAD_5MIN_BUNDLE__({ maxPairs: 10, limit: 100 }).");
})();
