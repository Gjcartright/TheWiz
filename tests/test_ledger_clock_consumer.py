"""Direct clock consumer checks; no operational imports."""

import pandas as pd
import pytest

from quant_platform.trade_ledger import build_trade_ledger


@pytest.mark.parametrize("datetime_index", [False, True])
def test_regular_mixed_iso_timestamps_preserve_exact_binding(datetime_index):
    timestamps = ["2026-01-01", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00+00:00"]
    index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC") if datetime_index else pd.RangeIndex(3)
    ledger = build_trade_ledger(pd.Series([1.0, 1.0, 0.0], index=index),
                                pd.Series([0.0, 0.01, 0.0], index=index), {},
                                timestamps=pd.Series(timestamps, index=index))
    assert ledger.closed_trades.iloc[0].net_pnl == pytest.approx(0.01)
    assert ledger.bar_ledger["timestamp"].tolist() == timestamps


def test_mixed_iso_timestamps_cannot_relabel_datetime_index():
    index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    with pytest.raises(ValueError, match="match the datetime ledger index"):
        build_trade_ledger(pd.Series([1.0, 1.0, 0.0], index=index),
                           pd.Series([0.0, 0.01, 0.0], index=index), {},
                           timestamps=["2026-01-02", "2026-01-02T01:00:00Z", "2026-01-02T02:00:00+00:00"])
