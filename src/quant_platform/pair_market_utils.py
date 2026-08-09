from __future__ import annotations

import re


def normalize_dydx_market(asset: str) -> str:
    """Return a normalized DYDX spot market key from an asset token.

    Examples:
    - ``BTC`` -> ``BTC-USD``
    - ``BTC-USD`` -> ``BTC-USD``
    - ``BTC/USD`` -> ``BTC-USD``
    """

    text = str(asset or "").upper().replace("/", "-").replace("_", "-").strip()
    parts = [part for part in re.split(r"[-_]", text) if part]
    if not parts:
        return ""
    if len(parts) == 1:
        return f"{parts[0]}-USD"
    if parts[-1] == "USD":
        return f"{parts[0]}-USD"
    return f"{parts[0]}-{parts[1]}"


def pair_markets_from_pair(pair: str) -> list[str]:
    """Parse a two-leg pair identifier into two DYDX market ids.

    Supports formats like ``AAA-USD-BBB-USD``, ``AAA/BBB``, ``AAA-BBB``,
    and mixed separators. Returns ``[AAA-USD, BBB-USD]`` when resolvable.
    """

    text = str(pair or "").upper().replace("_", "-").strip()
    if not text:
        return []

    if "-USD-" in text:
        left, right = text.split("-USD-", 1)
        if left and right:
            return [normalize_dydx_market(left), normalize_dydx_market(right)]

    # Fall back to generic token split for formats such as "AAA-BBB",
    # "AAA/BBB", or already normalized "AAA-USD" / "BBB-USD" forms.
    tokens = [token for token in re.split(r"[-_/]", text) if token and token != "USD"]
    if len(tokens) >= 2:
        return [normalize_dydx_market(tokens[0]), normalize_dydx_market(tokens[1])]

    return []
