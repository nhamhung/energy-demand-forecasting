"""Model pipeline construction, training, evaluation, and persistence.

Task: predict `Consumption` (Romania's national hourly electricity
demand, megawatts) from calendar features, lag/rolling-window features
of its own history, and *lagged* generation-mix features (see
`features.py`). Evaluated with MAE, RMSE, and MAPE.

**Why the train/test split is chronological, never shuffled or
k-folded on random rows**: see the original PJM version of this
project's reasoning, unchanged here — lag/rolling features are built
from each row's own recent past, so a random shuffle would leak
near-future information into training rows sitting next to shuffled-in
test rows.

**Why `MLPRegressor` is in `MODEL_FACTORIES`**: a lightweight,
zero-new-dependency stand-in for the neural network comparison a
related piece of prior work (a university project on a different
electricity-demand dataset) used — full deep learning frameworks
(TensorFlow/Keras, multiple architectures) are a much larger dependency
and build-time cost than this portfolio's other projects carry, so
`sklearn.neural_network.MLPRegressor` (already available via the
existing scikit-learn dependency) tests the same underlying question —
does a simple neural net beat gradient boosting here — without the
added weight.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf, adfuller

from . import config, features


def linear_estimator() -> BaseEstimator:
    return LinearRegression()


def random_forest_estimator() -> BaseEstimator:
    return RandomForestRegressor(n_estimators=300, max_depth=16, random_state=config.RANDOM_SEED, n_jobs=-1)


def lightgbm_estimator() -> BaseEstimator:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(n_estimators=300, random_state=config.RANDOM_SEED, verbosity=-1)


def mlp_estimator() -> BaseEstimator:
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        early_stopping=True,
        n_iter_no_change=10,
        max_iter=300,
        random_state=config.RANDOM_SEED,
    )


MODEL_FACTORIES: dict[str, "callable[[], BaseEstimator]"] = {
    "Linear Regression": linear_estimator,
    "Random Forest": random_forest_estimator,
    "LightGBM": lightgbm_estimator,
    "MLP (neural network)": mlp_estimator,
}


def build_pipeline(estimator: BaseEstimator | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", estimator if estimator is not None else lightgbm_estimator()),
        ]
    )


def chronological_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_fraction: float = config.TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_at = int(len(X) * (1 - test_fraction))
    return X.iloc[:split_at], X.iloc[split_at:], y.iloc[:split_at], y.iloc[split_at:]


def regression_scores(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def naive_seasonal_baseline(X: pd.DataFrame, y: pd.Series) -> dict:
    """"This hour's demand will be about what it was at the same hour
    last week" — no model at all, just `lag_168h` used directly.
    """
    return regression_scores(y, X["lag_168h"])


def cross_validate_pipeline(X: pd.DataFrame, y: pd.Series, estimator: BaseEstimator | None = None, cv: int = 5) -> dict:
    """`TimeSeriesSplit`, never `KFold` — see the module docstring."""
    splitter = TimeSeriesSplit(n_splits=cv)
    fold_scores = {"mae": [], "rmse": [], "mape": []}
    for train_idx, val_idx in splitter.split(X):
        pipeline = build_pipeline(estimator)
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = pipeline.predict(X.iloc[val_idx])
        scores = regression_scores(y.iloc[val_idx], preds)
        for key, value in scores.items():
            fold_scores[key].append(value)
    return {
        **{f"{k}_mean": np.mean(v) for k, v in fold_scores.items()},
        **{f"{k}_std": np.std(v) for k, v in fold_scores.items()},
    }


def train_pipeline(X: pd.DataFrame, y: pd.Series, estimator: BaseEstimator | None = None) -> Pipeline:
    pipeline = build_pipeline(estimator)
    pipeline.fit(X, y)
    return pipeline


def save_pipeline(pipeline: Pipeline, path: Path = config.MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path = config.MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train a model first: `python scripts/train.py`."
        )
    return joblib.load(path)


def stationarity_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test — the standard statistical check for
    whether a series is stationary (no trend/seasonality) before
    trusting a classical model like ARIMA on it directly. Returns the
    test statistic, p-value, and critical values.
    """
    statistic, p_value, used_lag, n_obs, critical_values, _ = adfuller(series, result_object=False)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "used_lag": used_lag,
        "n_obs": n_obs,
        "critical_values": critical_values,
    }


def autocorrelation(series: pd.Series, n_lags: int) -> np.ndarray:
    """Autocorrelation values for lags `0..n_lags`, used to justify
    which lag hours actually carry signal rather than picking `lag_24h`/
    `lag_168h` by convention alone.
    """
    return acf(series, nlags=n_lags, fft=True)


def recursive_forecast(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    anchor: pd.Timestamp,
    horizon_hours: int,
    seasonal_lookup: pd.Series | None = None,
    history_hours: int = 400,
) -> pd.Series:
    """Forecast `horizon_hours` hours forward from `anchor` (capped at
    `config.MAX_FORECAST_HORIZON_HOURS`), one hour at a time, feeding
    each prediction back in as input for the next hour.

    Every prediction beyond the first is genuinely recursive: hour 2's
    features depend in part on hour 1's *predicted* value, not a real
    one, and so on — errors can compound the further out this goes. This
    is a deliberate, visible property of multi-step forecasting, not a
    bug to hide. It's also why the horizon is capped at 24h: with
    `EXOGENOUS_LAG_HOURS` also at 24, every generation-mix feature stays
    grounded in real data for the whole window (its 24h-ago reference
    point never falls past `anchor`) — only the rolling-window features
    (which need the immediately preceding hours) ever read back a
    prediction instead of a real value.
    """
    if horizon_hours > config.MAX_FORECAST_HORIZON_HOURS:
        raise ValueError(
            f"horizon_hours={horizon_hours} exceeds the {config.MAX_FORECAST_HORIZON_HOURS}h cap "
            "(see config.py's module docstring for why)."
        )
    if anchor not in frame.index:
        raise ValueError(f"{anchor} is not in the given frame's index.")

    history = frame.loc[: anchor].iloc[-history_hours:].copy()
    predictions = {}

    for h in range(1, horizon_hours + 1):
        t = anchor + pd.Timedelta(hours=h)
        new_row = pd.DataFrame(
            {col: [history[col].iloc[-1]] for col in config.EXOGENOUS_COLS},
            index=pd.DatetimeIndex([t], name=history.index.name),
        )
        new_row[config.TARGET_COL] = np.nan  # placeholder — never read unshifted by build_feature_frame
        working = pd.concat([history, new_row])

        built = features.build_feature_frame(working, seasonal_lookup=seasonal_lookup)
        x_t = built.loc[[t]].drop(columns=[config.TARGET_COL])
        prediction = pipeline.predict(x_t)[0]

        predictions[t] = prediction
        history.loc[t] = new_row.iloc[0]
        history.loc[t, config.TARGET_COL] = prediction

    return pd.Series(predictions, name=config.TARGET_COL)


def forecast_error_by_horizon(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    anchors: list[pd.Timestamp],
    max_horizon: int = config.MAX_FORECAST_HORIZON_HOURS,
    seasonal_lookup: pd.Series | None = None,
) -> pd.DataFrame:
    """Run `recursive_forecast` from several different anchors and
    report mean absolute percentage error **per steps-ahead**, not
    averaged across the whole horizon — this is what actually shows
    whether/how much error compounds as the forecast reaches further
    into the future, which a single aggregate number would hide.
    """
    errors_by_step: dict[int, list[float]] = {h: [] for h in range(1, max_horizon + 1)}
    for anchor in anchors:
        last_needed = anchor + pd.Timedelta(hours=max_horizon)
        if last_needed not in frame.index:
            continue  # no real ground truth to compare against this far out
        forecast = recursive_forecast(pipeline, frame, anchor, max_horizon, seasonal_lookup=seasonal_lookup)
        actual = frame.loc[forecast.index, config.TARGET_COL]
        for h, (t, predicted) in enumerate(forecast.items(), start=1):
            errors_by_step[h].append(abs(predicted - actual.loc[t]) / actual.loc[t])

    rows = [
        {"steps_ahead": h, "mape": np.mean(errs), "n_anchors": len(errs)}
        for h, errs in errors_by_step.items()
        if errs
    ]
    return pd.DataFrame(rows)
