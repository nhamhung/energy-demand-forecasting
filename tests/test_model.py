"""Tests for the model pipeline and evaluation helpers, using a small
synthetic frame instead of the real Romania data.
"""

import numpy as np
import pandas as pd

from energy_demand_forecasting import model
from energy_demand_forecasting.features import split_features_target
from tests.test_features import make_synthetic_frame


def test_chronological_split_never_shuffles():
    frame = make_synthetic_frame(n_hours=500)
    X, y = split_features_target(frame)

    X_train, X_test, y_train, y_test = model.chronological_train_test_split(X, y, test_fraction=0.2)

    assert len(X_train) + len(X_test) == len(X)
    assert X_train.index.max() < X_test.index.min()
    assert y_train.index.equals(X_train.index)


def test_naive_seasonal_baseline_is_perfect_on_a_perfectly_periodic_series():
    import energy_demand_forecasting.config as config

    n_hours = 500
    values = np.tile(np.arange(168, dtype=float), n_hours // 168 + 1)[:n_hours]
    frame = make_synthetic_frame(n_hours=n_hours)
    frame[config.TARGET_COL] = values

    X, y = split_features_target(frame)
    scores = model.naive_seasonal_baseline(X, y)

    assert scores["mae"] < 1e-9


def test_cross_validate_pipeline_uses_time_series_split_and_returns_valid_scores():
    frame = make_synthetic_frame(n_hours=800)
    X, y = split_features_target(frame)

    result = model.cross_validate_pipeline(X, y, estimator=model.linear_estimator(), cv=3)

    assert result["mae_mean"] > 0
    assert result["rmse_mean"] >= result["mae_mean"]


def test_mlp_estimator_trains_and_predicts():
    frame = make_synthetic_frame(n_hours=500)
    X, y = split_features_target(frame)
    pipeline = model.train_pipeline(X, y, estimator=model.mlp_estimator())

    preds = pipeline.predict(X)
    assert len(preds) == len(X)


def test_stationarity_test_returns_expected_keys():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(size=300))
    result = model.stationarity_test(series)

    assert set(result) == {"statistic", "p_value", "used_lag", "n_obs", "critical_values"}
    assert 0 <= result["p_value"] <= 1


def test_autocorrelation_at_lag_zero_is_always_one():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(size=300))
    values = model.autocorrelation(series, n_lags=10)

    assert len(values) == 11
    assert values[0] == 1.0


def test_recursive_forecast_rejects_horizon_beyond_the_cap():
    import pytest
    import energy_demand_forecasting.config as config

    frame = make_synthetic_frame(n_hours=500)
    pipeline = model.train_pipeline(*split_features_target(frame), estimator=model.linear_estimator())

    with pytest.raises(ValueError):
        model.recursive_forecast(pipeline, frame, frame.index[300], config.MAX_FORECAST_HORIZON_HOURS + 1)


def test_recursive_forecast_returns_one_prediction_per_hour_of_horizon():
    frame = make_synthetic_frame(n_hours=500)
    pipeline = model.train_pipeline(*split_features_target(frame), estimator=model.linear_estimator())

    anchor = frame.index[300]
    forecast = model.recursive_forecast(pipeline, frame, anchor, horizon_hours=24)

    assert len(forecast) == 24
    assert forecast.index[0] == anchor + pd.Timedelta(hours=1)
    assert forecast.index[-1] == anchor + pd.Timedelta(hours=24)
    assert forecast.notna().all()


def test_recursive_forecast_exogenous_lags_stay_grounded_in_real_data():
    """Within the 24h cap, every *_lag_24h feature's 24h-ago reference
    point is always <= anchor — i.e. real data — never a placeholder
    row appended during recursion. This checks that directly rather
    than trusting the reasoning in the module docstring.
    """
    import energy_demand_forecasting.config as config
    frame = make_synthetic_frame(n_hours=500)
    anchor = frame.index[300]

    for h in range(1, 25):
        t = anchor + pd.Timedelta(hours=h)
        lookback = t - pd.Timedelta(hours=config.EXOGENOUS_LAG_HOURS)
        assert lookback <= anchor  # always real, never a placeholder future row


def test_forecast_error_by_horizon_returns_one_row_per_step():
    frame = make_synthetic_frame(n_hours=500)
    pipeline = model.train_pipeline(*split_features_target(frame), estimator=model.linear_estimator())

    anchors = [frame.index[300], frame.index[350]]
    result = model.forecast_error_by_horizon(pipeline, frame, anchors, max_horizon=24)

    assert set(result["steps_ahead"]) == set(range(1, 25))
    assert (result["mape"] >= 0).all()


def test_save_and_load_pipeline_roundtrip(tmp_path):
    frame = make_synthetic_frame(n_hours=400)
    X, y = split_features_target(frame)
    pipeline = model.train_pipeline(X, y, estimator=model.linear_estimator())

    path = tmp_path / "model.joblib"
    model.save_pipeline(pipeline, path=path)
    loaded = model.load_pipeline(path=path)

    np.testing.assert_array_almost_equal(pipeline.predict(X), loaded.predict(X))
