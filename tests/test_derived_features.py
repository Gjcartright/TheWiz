import pandas as pd

from quant_platform.derived_features import add_derived_beta_from_prices


def test_add_derived_beta_from_prices_requires_enough_return_history():
    frame = pd.DataFrame({"price_x": [100.0, 101.0, 102.0], "price_y": [50.0, 50.5, 51.0]})

    enriched = add_derived_beta_from_prices(frame)

    assert "beta" not in enriched.columns


def test_add_derived_beta_from_prices_adds_source_when_native_beta_missing():
    frame = pd.DataFrame(
        {
            "price_x": [100.0, 102.0, 101.0, 104.0, 103.0],
            "price_y": [50.0, 51.0, 50.0, 52.0, 51.5],
        }
    )

    enriched = add_derived_beta_from_prices(frame)

    assert "beta" in enriched.columns
    assert enriched["beta"].notna().all()
    assert set(enriched["beta_source"]) == {"derived_from_price_returns"}


def test_add_derived_beta_from_prices_preserves_native_beta():
    frame = pd.DataFrame(
        {
            "price_x": [100.0, 102.0, 101.0, 104.0],
            "price_y": [50.0, 51.0, 50.0, 52.0],
            "beta": [0.8, 0.8, 0.8, 0.8],
        }
    )

    enriched = add_derived_beta_from_prices(frame)

    assert list(enriched["beta"]) == [0.8, 0.8, 0.8, 0.8]
    assert "beta_source" not in enriched.columns
