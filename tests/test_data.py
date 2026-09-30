"""Tests for the DST/outage cleaning logic in `data.clean_frame` — the
central data-quality decision this project makes (see `config.py`'s
module docstring).
"""

import pandas as pd

from energy_demand_forecasting import config, data


def _make_raw(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df[config.DATETIME_COL] = pd.to_datetime(df[config.DATETIME_COL])
    for col in config.EXOGENOUS_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def test_duplicate_timestamps_are_averaged():
    df = _make_raw(
        {
            config.DATETIME_COL: ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 01:00", "2020-01-01 02:00"],
            config.TARGET_COL: [100.0, 90.0, 110.0, 120.0],
        }
    )
    cleaned = data.clean_frame(df)

    assert cleaned.loc["2020-01-01 01:00", config.TARGET_COL] == 100.0
    assert len(cleaned) == 3


def test_missing_hours_are_interpolated():
    df = _make_raw(
        {
            config.DATETIME_COL: ["2020-01-01 00:00", "2020-01-01 02:00"],
            config.TARGET_COL: [100.0, 120.0],
        }
    )
    cleaned = data.clean_frame(df)

    assert len(cleaned) == 3
    assert cleaned.loc["2020-01-01 01:00", config.TARGET_COL] == 110.0
    assert not cleaned.isna().any().any()


def test_gap_lengths_distinguishes_isolated_from_long_runs():
    df = _make_raw(
        {
            # 00:00 present, 01:00-03:00 missing (a 3-hour run), 04:00 present, 06:00 present (05:00 an isolated gap)
            config.DATETIME_COL: ["2020-01-01 00:00", "2020-01-01 04:00", "2020-01-01 06:00"],
            config.TARGET_COL: [100.0, 100.0, 100.0],
        }
    )
    runs = data.gap_lengths(df)

    assert sorted(runs.tolist()) == [1, 3]


def test_all_numeric_columns_survive_cleaning():
    df = _make_raw(
        {
            config.DATETIME_COL: ["2020-01-01 00:00", "2020-01-01 01:00"],
            config.TARGET_COL: [100.0, 110.0],
            "Wind": [10.0, 12.0],
        }
    )
    cleaned = data.clean_frame(df)

    assert set(cleaned.columns) == {config.TARGET_COL, *config.EXOGENOUS_COLS}
