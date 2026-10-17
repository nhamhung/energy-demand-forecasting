"""Tests for the DST/outage cleaning logic in `data.clean_frame` — the
central data-quality decision this project makes (see `config.py`'s
module docstring).
"""

import numpy as np
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


def _synthetic_hourly_raw(n_days: int = 21, seed: int = 0) -> pd.DataFrame:
    """Three weeks of realistic-looking hourly rows: a daily demand cycle
    (low at night, high in the evening), a near-constant nuclear output,
    and other columns that move independently of demand."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=24 * n_days, freq="h")
    hours = np.arange(len(index))
    demand = 6500 - 1000 * np.cos(2 * np.pi * (hours % 24 - 3) / 24) + rng.normal(0, 60, len(index))
    df = pd.DataFrame({config.DATETIME_COL: index, config.TARGET_COL: demand.round()})
    df["Nuclear"] = 1380 + rng.normal(0, 5, len(index)).round()
    for col in ["Wind", "Hydroelectric", "Oil and Gas", "Coal", "Solar", "Biomass"]:
        df[col] = rng.uniform(100, 2000, len(index)).round()
    df["Production"] = df[["Nuclear", "Wind", "Hydroelectric", "Oil and Gas", "Coal", "Solar", "Biomass"]].sum(axis=1)
    return df


def _inject_scaled_copy_night(df: pd.DataFrame, start: str, factors: list[float]) -> pd.DataFrame:
    """Reproduce the real feed fault: from `start` on, each row is the
    previous row with *every* numeric column scaled by one factor."""
    df = df.copy()
    numeric = [config.TARGET_COL] + config.EXOGENOUS_COLS
    pos = df.index[df[config.DATETIME_COL] == pd.Timestamp(start)][0]
    for i, factor in enumerate(factors):
        df.loc[pos + i, numeric] = (df.loc[pos + i - 1, numeric] * factor).round().values
    return df


def test_no_hour_is_flagged_in_clean_data():
    df = _synthetic_hourly_raw()
    regular = df.set_index(config.DATETIME_COL)

    assert not data.flag_implausible_readings(regular).any()
    cleaned = data.clean_frame(df)
    np.testing.assert_array_equal(cleaned.values, regular[cleaned.columns].values)


def test_scaled_copy_night_is_flagged_end_to_end_and_interpolated():
    df = _synthetic_hourly_raw()
    # 10 fabricated hours from 20:00: copies of 19:00 growing ~7%/hour, so
    # night "demand" ends far above the evening peak (like 2025-11-27).
    bad = _inject_scaled_copy_night(df, "2025-01-10 20:00", [1.0, 1.04] + [1.07] * 8)
    flags = data.flag_implausible_readings(bad.set_index(config.DATETIME_COL))

    expected = pd.date_range("2025-01-10 20:00", periods=10, freq="h")
    assert list(flags[flags].index) == list(expected)          # the whole run, and nothing else
    assert not flags.loc["2025-01-10 19:00"]                   # the genuine hour that was copied is kept

    cleaned = data.clean_frame(bad)
    before, after = cleaned.loc["2025-01-10 19:00"], cleaned.loc["2025-01-11 06:00"]
    for i, t in enumerate(expected, start=1):
        assert np.isclose(cleaned.loc[t, config.TARGET_COL],
                          before[config.TARGET_COL] + (after[config.TARGET_COL] - before[config.TARGET_COL]) * i / 11)
    assert cleaned[config.TARGET_COL].max() < df[config.TARGET_COL].max() + 1


def test_physically_impossible_nuclear_triggers_a_mild_copy_run():
    df = _synthetic_hourly_raw()
    # A milder copy run that stays under the 1.5x demand trigger but pushes
    # nuclear output above what the plant can produce.
    bad = _inject_scaled_copy_night(df, "2025-01-12 18:00", [1.03, 1.04, 1.05])
    flags = data.flag_implausible_readings(bad.set_index(config.DATETIME_COL))

    assert bad.set_index(config.DATETIME_COL).loc["2025-01-12 20:00", "Nuclear"] > config.NUCLEAR_MAX_PLAUSIBLE_MW
    assert list(flags[flags].index) == list(pd.date_range("2025-01-12 18:00", periods=3, freq="h"))


def test_a_single_unusual_but_real_hour_is_not_flagged():
    df = _synthetic_hourly_raw()
    # +30% demand for one hour, other columns moving independently: unusual,
    # below the trigger, and not a proportional copy, so it must be kept.
    pos = df.index[df[config.DATETIME_COL] == pd.Timestamp("2025-01-08 21:00")][0]
    df.loc[pos, config.TARGET_COL] *= 1.3
    # A holiday-like collapse (demand halves) is never flagged either.
    low = df[config.DATETIME_COL].between("2025-01-15 00:00", "2025-01-15 23:00")
    df.loc[low, config.TARGET_COL] *= 0.45

    assert not data.flag_implausible_readings(df.set_index(config.DATETIME_COL)).any()


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
