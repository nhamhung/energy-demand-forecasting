"""Feature engineering: calendar cycles, lag/rolling-window features
from the target's own history, lagged exogenous generation-mix
features, and an optional, leakage-safe seasonal-decomposition feature.

Single source of truth for the notebook, `scripts/train.py`, and the
Streamlit app.
"""

import numpy as np
import pandas as pd
from holidays import RO as ro_holidays_factory
from statsmodels.tsa.seasonal import seasonal_decompose

from . import config


def _cyclical(values: pd.Series, period: int, name: str) -> pd.DataFrame:
    angle = 2 * np.pi * values / period
    return pd.DataFrame({f"{name}_sin": np.sin(angle), f"{name}_cos": np.cos(angle)})


def fit_seasonal_lookup(train_target: pd.Series, period: int = 24) -> pd.Series:
    """Fit an additive seasonal decomposition **on the training split
    only**, and return just its periodic seasonal component as an
    `hour -> offset` lookup (24 values, one per hour-of-day).

    This is a deliberate departure from the more common pattern of
    running `statsmodels.tsa.seasonal.seasonal_decompose` on the *whole*
    series before splitting into train/test, which this project's
    reference material does. That's leakage: the decomposition's trend
    component is a **centered** moving average, computed from points
    both before *and after* each timestamp — including the trend or
    residual from a decomposition fit on the full series in a model's
    features means the training features are quietly built with
    information from the future test period. Fitting only on the
    training split, and reusing only the strictly periodic seasonal
    component (24 fixed hour-of-day offsets, safe to reapply to any
    future timestamp without recomputation), avoids this while still
    testing the same underlying idea.
    """
    result = seasonal_decompose(train_target, model="additive", period=period, extrapolate_trend="period")
    seasonal = result.seasonal.iloc[:period]
    seasonal.index = range(period)
    return seasonal


def build_feature_frame(
    frame: pd.DataFrame, seasonal_lookup: pd.Series | None = None
) -> pd.DataFrame:
    """Build calendar + lag + rolling-window + lagged-exogenous features
    from a clean, regular hourly frame (see `data.load_clean_frame`).

    `seasonal_lookup` is optional and, if given, must come from
    `fit_seasonal_lookup` run on a **training** split — see that
    function's docstring for why it can't be safely computed from the
    full frame passed in here.
    """
    df = frame.copy()

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    ro_holidays = ro_holidays_factory(years=range(df.index.year.min(), df.index.year.max() + 1))
    holiday_dates = set(ro_holidays.keys())
    df["is_holiday"] = df.index.normalize().to_series().dt.date.isin(holiday_dates).astype(int).values

    df = pd.concat(
        [
            df,
            _cyclical(df["hour"], 24, "hour"),
            _cyclical(df["dayofweek"], 7, "dayofweek"),
            _cyclical(df["month"], 12, "month"),
        ],
        axis=1,
    )

    for lag in config.LAG_HOURS:
        df[f"lag_{lag}h"] = df[config.TARGET_COL].shift(lag)

    for window in config.ROLLING_WINDOWS_HOURS:
        shifted = df[config.TARGET_COL].shift(1)
        df[f"rolling_mean_{window}h"] = shifted.rolling(window=window).mean()
        df[f"rolling_std_{window}h"] = shifted.rolling(window=window).std()

    # Lagged exogenous generation-mix features — never unlagged (see
    # config.py's module docstring for the leakage reasoning).
    exo_lag = config.EXOGENOUS_LAG_HOURS
    for col in config.EXOGENOUS_COLS:
        df[f"{col.replace(' ', '_')}_lag_{exo_lag}h"] = df[col].shift(exo_lag)

    df[f"net_export_lag_{exo_lag}h"] = (
        df["Production"].shift(exo_lag) - df[config.TARGET_COL].shift(exo_lag)
    )
    renewable_sum = sum(df[col].shift(exo_lag) for col in config.RENEWABLE_COLS)
    df[f"renewable_share_lag_{exo_lag}h"] = renewable_sum / df["Production"].shift(exo_lag)

    df = df.drop(columns=config.EXOGENOUS_COLS)

    if seasonal_lookup is not None:
        df["seasonal_24h"] = df["hour"].map(seasonal_lookup)

    return df


def split_features_target(
    frame: pd.DataFrame, seasonal_lookup: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Build features, then drop the warm-up rows where a lag/rolling
    feature isn't available yet."""
    built = build_feature_frame(frame, seasonal_lookup=seasonal_lookup).dropna()
    y = built[config.TARGET_COL]
    X = built.drop(columns=[config.TARGET_COL])
    return X, y
