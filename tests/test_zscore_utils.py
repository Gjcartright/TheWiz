from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.strategies import zscore_signal
from quant_platform.zscore_utils import coalesce_zscore


def test_coalesce_zscore_preserves_valid_all_zero_preferred_series():
    frame = pd.DataFrame(
        {
            "zscore_reconstructed": [0.0, 0.0, 0.0],
            "zscore": [2.5, 2.5, 2.5],
            "rolling_zscore": [-2.5, -2.5, -2.5],
        }
    )

    result = coalesce_zscore(frame)

    assert result.tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("prefix", [[0.0, 0.0], [np.nan, np.nan], [0.0, np.nan]])
def test_selected_source_does_not_change_when_future_values_arrive(prefix):
    frame = pd.DataFrame({"zscore_reconstructed": prefix, "zscore": [3.0, -3.0]})
    extended = pd.concat(
        [frame, pd.DataFrame({"zscore_reconstructed": [-4.0], "zscore": [1.0]})],
        ignore_index=True,
    )

    np.testing.assert_allclose(
        coalesce_zscore(frame), coalesce_zscore(extended).iloc[:2], equal_nan=True
    )
    pd.testing.assert_series_equal(zscore_signal(frame), zscore_signal(extended).iloc[:2])
    assert (zscore_signal(frame) == 0).all()


def test_explicit_source_binding_survives_future_schema_growth():
    prefix = pd.DataFrame({"zscore": [3.0, -3.0]})
    extended = pd.concat(
        [prefix, pd.DataFrame({"zscore": [1.0], "zscore_reconstructed": [0.0]})],
        ignore_index=True,
    )
    expected = coalesce_zscore(prefix, source_column="zscore")
    actual = coalesce_zscore(extended, source_column="zscore")

    np.testing.assert_allclose(expected, actual.iloc[:2], equal_nan=True)
    assert actual.attrs["zscore_source_column"] == "zscore"
    prefix.attrs["zscore_source_column"] = "zscore"
    extended.attrs["zscore_source_column"] = "zscore"
    pd.testing.assert_series_equal(zscore_signal(prefix), zscore_signal(extended).iloc[:2])
    with pytest.raises(ValueError, match="conflicts"):
        coalesce_zscore(extended, source_column="zscore_reconstructed")


def test_missing_bound_source_does_not_substitute_another_metric():
    result = coalesce_zscore(
        pd.DataFrame({"zscore": [3.0, 2.0]}),
        source_column="zscore_reconstructed",
    )
    assert result.isna().all()
