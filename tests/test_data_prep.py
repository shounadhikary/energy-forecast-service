import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from data_prep import add_time_features, add_lag_features, build_features


def make_sample_df():
    """Create a small fake dataset for testing."""
    times = pd.date_range("2020-01-01", periods=300, freq="h")
    return pd.DataFrame({
        "Datetime": times,
        "consumption": range(300),
    })


def test_time_features_exist():
    df = add_time_features(make_sample_df())
    for col in ["hour", "dayofweek", "month", "dayofyear", "is_weekend"]:
        assert col in df.columns


def test_is_weekend_values():
    df = add_time_features(make_sample_df())
    assert set(df["is_weekend"].unique()).issubset({0, 1})


def test_lag_features_exist():
    df = add_lag_features(make_sample_df())
    for col in ["lag_1h", "lag_24h", "lag_168h"]:
        assert col in df.columns


def test_build_features_no_nans():
    df = build_features(make_sample_df())
    assert df.isna().sum().sum() == 0
    assert len(df) > 0