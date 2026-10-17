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


def test_persistence_baseline_uses_the_previous_hour():
    import energy_demand_forecasting.config as config

    frame = make_synthetic_frame(n_hours=500)
    X, y = split_features_target(frame)
    _, X_test, _, y_test = model.chronological_train_test_split(X, y, test_fraction=0.2)

    scores = model.persistence_baseline(frame, y_test)
    previous = frame[config.TARGET_COL].shift(1).loc[y_test.index]
    assert np.isclose(scores["mae"], (y_test - previous).abs().mean())
    # The very first test hour is scored against the last training hour, not dropped.
    assert previous.notna().all()


def test_persistence_baseline_is_perfect_on_a_constant_series():
    import energy_demand_forecasting.config as config

    frame = make_synthetic_frame(n_hours=500)
    frame[config.TARGET_COL] = 5000.0
    _, y = split_features_target(frame)

    assert model.persistence_baseline(frame, y)["mae"] == 0.0


def test_cross_validation_only_touches_the_rows_it_is_given():
    frame = make_synthetic_frame(n_hours=800)
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = model.chronological_train_test_split(X, y, test_fraction=0.25)

    folds = model.time_series_folds(X_train, cv=3)
    touched = np.unique(np.concatenate([np.concatenate([tr, va]) for tr, va in folds]))
    assert touched.max() < len(X_train)                         # positions within X_train only
    assert X_train.index[folds[-1][1][-1]] == X_train.index[-1]  # last fold ends where training ends
    assert X_train.index[-1] < X_test.index.min()               # ...strictly before the test period

    # Scrambling the test period cannot change cross-validation on the training period.
    scores = model.cross_validate_pipeline(X_train, y_train, estimator=model.linear_estimator(), cv=3)
    X_scrambled, y_scrambled = X.copy(), y.copy()
    X_scrambled.loc[X_test.index] = 0.0
    y_scrambled.loc[y_test.index] = 1e9
    X_train_2, _, y_train_2, _ = model.chronological_train_test_split(X_scrambled, y_scrambled, test_fraction=0.25)
    assert model.cross_validate_pipeline(X_train_2, y_train_2, estimator=model.linear_estimator(), cv=3) == scores


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
    fitted = model.train_pipeline(*split_features_target(frame), estimator=model.linear_estimator())

    seen_rows = []

    class Recorder:
        """Passes predictions through, keeping every feature row the forecast feeds the model."""

        def predict(self, x):
            seen_rows.append(x)
            return fitted.predict(x)

    forecast = model.recursive_forecast(Recorder(), frame, anchor, horizon_hours=24)

    lag = config.EXOGENOUS_LAG_HOURS
    for x in seen_rows:
        t = x.index[0]
        real = frame.loc[t - pd.Timedelta(hours=lag)]
        for col in config.EXOGENOUS_COLS:
            assert x[f"{col.replace(' ', '_')}_lag_{lag}h"].iloc[0] == real[col]
        assert x[f"net_export_lag_{lag}h"].iloc[0] == real["Production"] - real[config.TARGET_COL]

    # Nothing after the anchor may influence the forecast: scramble the future, same result.
    poisoned = frame.copy()
    poisoned.loc[poisoned.index > anchor] = poisoned.loc[poisoned.index > anchor] * 10 + 12345
    pd.testing.assert_series_equal(model.recursive_forecast(fitted, poisoned, anchor, horizon_hours=24), forecast)


def test_forecast_error_by_horizon_returns_one_row_per_step():
    frame = make_synthetic_frame(n_hours=500)
    pipeline = model.train_pipeline(*split_features_target(frame), estimator=model.linear_estimator())

    anchors = [frame.index[300], frame.index[350]]
    result = model.forecast_error_by_horizon(pipeline, frame, anchors, max_horizon=24)

    assert set(result["steps_ahead"]) == set(range(1, 25))
    assert (result[["mape", "persistence_mape", "seasonal_naive_mape"]] >= 0).all().all()

    # Baselines use only what is known at the anchor: the last known value, and t - 168h.
    import energy_demand_forecasting.config as config
    c = frame[config.TARGET_COL]
    H = pd.Timedelta(hours=1)
    step_5 = result.set_index("steps_ahead").loc[5]
    expected_persistence = np.mean([abs(c[a] - c[a + 5 * H]) / c[a + 5 * H] for a in anchors])
    expected_seasonal = np.mean([abs(c[a + 5 * H - 168 * H] - c[a + 5 * H]) / c[a + 5 * H] for a in anchors])
    assert np.isclose(step_5["persistence_mape"], expected_persistence)
    assert np.isclose(step_5["seasonal_naive_mape"], expected_seasonal)


def test_save_and_load_pipeline_roundtrip(tmp_path):
    frame = make_synthetic_frame(n_hours=400)
    X, y = split_features_target(frame)
    pipeline = model.train_pipeline(X, y, estimator=model.linear_estimator())

    path = tmp_path / "model.joblib"
    model.save_pipeline(pipeline, path=path)
    loaded = model.load_pipeline(path=path)

    np.testing.assert_array_almost_equal(pipeline.predict(X), loaded.predict(X))
