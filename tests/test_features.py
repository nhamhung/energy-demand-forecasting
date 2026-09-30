"""Tests for feature engineering, using a small synthetic hourly frame
instead of the real ~62k-row Romania data.
"""

import numpy as np
import pandas as pd

from energy_demand_forecasting import config
from energy_demand_forecasting.features import build_feature_frame, fit_seasonal_lookup, split_features_target


def make_synthetic_frame(n_hours: int = 400, seed: int = 0) -> pd.DataFrame:
    """A plain hourly multivariate frame with a daily sine pattern plus
    noise for Consumption, and independent series for each exogenous
    column — mirrors the real data's shape without needing the real
    62k-row file.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    hours = np.arange(n_hours)
    consumption = 6000 + 1000 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 50, n_hours)

    df = pd.DataFrame({config.TARGET_COL: consumption}, index=index)
    for col in config.EXOGENOUS_COLS:
        df[col] = rng.uniform(100, 2000, n_hours)
    df["Production"] = df[config.TARGET_COL] + rng.normal(0, 50, n_hours)
    return df


def test_cyclical_hour_encoding_wraps_around():
    frame = make_synthetic_frame(n_hours=48)
    built = build_feature_frame(frame)

    hour_23 = built[built["hour"] == 23].iloc[0]
    hour_0 = built[built["hour"] == 0].iloc[0]
    dist_adjacent = np.hypot(hour_23["hour_sin"] - hour_0["hour_sin"], hour_23["hour_cos"] - hour_0["hour_cos"])

    hour_12 = built[built["hour"] == 12].iloc[0]
    dist_far = np.hypot(hour_23["hour_sin"] - hour_12["hour_sin"], hour_23["hour_cos"] - hour_12["hour_cos"])

    assert dist_adjacent < dist_far


def test_lag_features_shift_correctly():
    frame = make_synthetic_frame(n_hours=400)
    built = build_feature_frame(frame)

    non_null = built.dropna(subset=["lag_24h"])
    t = non_null.index[100]
    assert non_null.loc[t, "lag_24h"] == frame.loc[t - pd.Timedelta(hours=24), config.TARGET_COL]


def test_rolling_features_never_include_the_current_hour():
    n_hours = 400
    index = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    frame = make_synthetic_frame(n_hours=n_hours)
    frame[config.TARGET_COL] = 100.0
    spike_position = 300
    frame.iloc[spike_position, frame.columns.get_loc(config.TARGET_COL)] = 100_000.0

    built = build_feature_frame(frame)
    spike_time = index[spike_position]
    assert built.loc[spike_time, "rolling_mean_24h"] == 100.0


def test_exogenous_features_are_lagged_not_contemporaneous():
    frame = make_synthetic_frame(n_hours=400)
    built = build_feature_frame(frame)

    non_null = built.dropna(subset=["Wind_lag_24h"])
    t = non_null.index[200]
    assert non_null.loc[t, "Wind_lag_24h"] == frame.loc[t - pd.Timedelta(hours=24), "Wind"]
    # The raw, unlagged exogenous columns must never appear as features.
    for col in config.EXOGENOUS_COLS:
        assert col not in built.columns


def test_renewable_share_is_bounded_between_zero_and_one():
    frame = make_synthetic_frame(n_hours=400)
    built = build_feature_frame(frame).dropna(subset=["renewable_share_lag_24h"])

    assert (built["renewable_share_lag_24h"] >= 0).all()


def test_seasonal_lookup_has_one_value_per_hour_and_applies_without_lookahead():
    frame = make_synthetic_frame(n_hours=800)
    train = frame.iloc[:600]
    lookup = fit_seasonal_lookup(train[config.TARGET_COL])

    assert set(lookup.index) == set(range(24))

    built = build_feature_frame(frame, seasonal_lookup=lookup)
    # Every row's seasonal_24h must equal the lookup value for its hour,
    # for rows both inside and outside the training window used to fit it.
    for t in [frame.index[50], frame.index[700]]:
        assert built.loc[t, "seasonal_24h"] == lookup[t.hour]


def test_split_features_target_drops_warmup_rows_and_aligns_shapes():
    frame = make_synthetic_frame(n_hours=400)
    X, y = split_features_target(frame)

    assert len(X) == len(y)
    assert not X.isna().any().any()
    assert config.TARGET_COL not in X.columns
    assert len(X) < len(frame)
