"""Data loading and cleaning.

The raw CSV is not committed to the repo (best fetched fresh rather than
duplicated here). Download it first — see the project README — or let
`_require_file` fetch it automatically via the Kaggle API (used when
deploying without a Docker image that already bakes the file in; see
`app/pages_src/shared.py` for how deployed credentials get wired in).
Remember this source is live and continuously updated (see the module
docstring in `config.py`) — an automatic fetch at deploy time will pull
whatever is current then, not the exact snapshot this project's numbers
were computed from.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

_KAGGLE_EXTRACTED_NAME = "electricityConsumptionAndProductioction.csv"


def _download_from_kaggle() -> bool:
    """Best-effort automatic fetch via the Kaggle API, including the
    rename this dataset's extracted filename needs (see
    `_KAGGLE_EXTRACTED_NAME`). Returns whether the target file exists
    afterward. Silently does nothing (returns False) if the `kaggle`
    package isn't installed or no credentials are configured — callers
    fall back to the manual-download error message either way.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(config.KAGGLE_DATASET, path=str(config.DATA_RAW_DIR), unzip=True, quiet=True)
        extracted = config.DATA_RAW_DIR / _KAGGLE_EXTRACTED_NAME
        if extracted.exists() and not config.RAW_CSV.exists():
            extracted.rename(config.RAW_CSV)
    except Exception:
        return False
    return config.RAW_CSV.exists()


def _require_file(path: Path) -> Path:
    if not path.exists():
        _download_from_kaggle()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found, and automatic download via the Kaggle API "
            "didn't produce it either (no credentials configured, or the "
            "`kaggle` package isn't installed). Download the dataset "
            "manually instead — see the README's 'Get the data' section, e.g.:\n"
            f"  kaggle datasets download -d {config.KAGGLE_DATASET} -p {config.DATA_RAW_DIR}\n"
            f"  unzip -o {config.DATA_RAW_DIR / 'hourly-electricity-consumption-and-production.zip'} -d {config.DATA_RAW_DIR}\n"
            f"  mv {config.DATA_RAW_DIR / _KAGGLE_EXTRACTED_NAME} {config.RAW_CSV}"
        )
    return path


def using_sample_data() -> bool:
    """Whether the bundled Kaggle-derived one-year sample is active."""
    return config.SAMPLE_CSV.exists() and os.getenv("USE_FULL_KAGGLE_DATA", "").lower() not in {"1", "true", "yes"}


def load_raw() -> pd.DataFrame:
    """Load the CSV exactly as published, only parsing the datetime
    column — no cleaning yet. Useful for demonstrating the DST/outage
    issues directly against the untouched data (see `config.py`'s
    docstring).
    """
    path = config.SAMPLE_CSV if using_sample_data() else _require_file(config.RAW_CSV)
    df = pd.read_csv(path)
    df[config.DATETIME_COL] = pd.to_datetime(df[config.DATETIME_COL])
    return df.sort_values(config.DATETIME_COL).reset_index(drop=True)


def _regular_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Average duplicate timestamps and reindex onto a strictly regular
    hourly grid, leaving missing hours as NaN (no interpolation yet)."""
    numeric_cols = [config.TARGET_COL] + config.EXOGENOUS_COLS
    grouped = df.groupby(config.DATETIME_COL)[numeric_cols].mean()
    full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq="h")
    grouped = grouped.reindex(full_index)
    grouped.index.name = config.DATETIME_COL
    return grouped


def _same_hour_reference(target: pd.Series) -> pd.Series:
    """Median of the same clock hour on the `SPIKE_REFERENCE_DAYS` days
    before and after each hour (the hour itself excluded). A single bad
    night only contributes one of these values, so the median stays a
    clean "what this hour normally looks like this week" reference."""
    days = config.SPIKE_REFERENCE_DAYS
    offsets = [24 * k for k in range(-days, days + 1) if k != 0]
    neighbours = pd.concat([target.shift(o) for o in offsets], axis=1)
    reference = neighbours.median(axis=1, skipna=True)
    return reference.where(neighbours.notna().sum(axis=1) >= config.SPIKE_MIN_REFERENCE_DAYS)


def flag_implausible_readings(frame: pd.DataFrame) -> pd.Series:
    """Flag the fabricated night-time stretches described in `config.py`
    (a boolean Series aligned with `frame.index`, True = do not trust).

    `frame` must be strictly hourly (see `_regular_hourly`); NaN rows are
    allowed and are never flagged.

    Why this shape of rule: the bad readings are **not** isolated
    one-hour spikes. On each bad night the feed repeats the last genuine
    hour's whole row (every generation column and `Consumption`) scaled
    by a slowly growing factor, for 9-14 hours, until real readings
    resume and the value drops by thousands of MW in one hour. So:

    1. **Trigger** (conservative — only clearly impossible values):
       a. `Consumption` above `SPIKE_RATIO_THRESHOLD` (1.5) times the
          median of the same clock hour over the surrounding +/-
          `SPIKE_REFERENCE_DAYS` days. A same-hour reference is used
          instead of a plain rolling median because demand swings
          ~2,000 MW within every day and a 9-14 hour bad stretch would
          contaminate a short centred window. Low values are never
          flagged (real holidays go below 0.5x).
       b. `Nuclear` above `NUCLEAR_MAX_PLAUSIBLE_MW` (1,500 MW), more
          than Romania's only nuclear plant can produce and above every
          2019-2024 reading — only a scaled-up copied row gets there.
    2. **Extent**: each triggered hour's flag is extended backwards and
       forwards over the contiguous run of hours whose `COPY_CHECK_COLS`
       all changed by the same factor (within `COPY_RATIO_TOLERANCE`)
       relative to the hour before. Real hours essentially never do
       that — nuclear output, for one, is nearly constant while demand
       moves — so the extension only ever covers fabricated copies. The
       hour each run copies from (the last genuine reading) is kept.

    Milder stretches of the same fault that never reach either trigger,
    and older flat-lined (repeated) readings, are deliberately left
    alone: the rule is meant to remove readings that are clearly
    impossible, not to second-guess every unusual hour.
    """
    target = frame[config.TARGET_COL]
    triggered = (target / _same_hour_reference(target)) > config.SPIKE_RATIO_THRESHOLD
    if "Nuclear" in frame.columns:
        triggered |= frame["Nuclear"] > config.NUCLEAR_MAX_PLAUSIBLE_MW

    cols = [col for col in config.COPY_CHECK_COLS if col in frame.columns]
    step_ratio = frame[cols] / frame[cols].shift(1)
    is_copy = (step_ratio.max(axis=1) - step_ratio.min(axis=1)) < config.COPY_RATIO_TOLERANCE
    is_copy &= step_ratio.notna().all(axis=1)

    run_id = (is_copy != is_copy.shift()).cumsum()
    copy_run_triggered = triggered.groupby(run_id).transform("any") & is_copy
    return (triggered | copy_run_triggered).fillna(False).astype(bool)


def implausible_readings(df: pd.DataFrame) -> pd.DataFrame:
    """The raw readings `clean_frame` discards as implausible, with the
    same-hour reference they were judged against — for reporting."""
    regular = _regular_hourly(df)
    flags = flag_implausible_readings(regular)
    out = regular.loc[flags, [config.TARGET_COL, "Production", "Nuclear"]].copy()
    out["same_hour_median"] = _same_hour_reference(regular[config.TARGET_COL]).loc[flags]
    out["ratio"] = out[config.TARGET_COL] / out["same_hour_median"]
    return out


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the data-quality issues found by checking the raw file
    directly (see `config.py`'s module docstring):

    1. **Duplicate fall-back timestamps**: averaged into one reading
       per timestamp, across every numeric column — there's no way to
       tell which of the readings is "more correct" from this file
       alone, so averaging is the least-assuming choice.
    2. **Implausible fabricated readings** (a few 2025-2026 nights where
       demand climbs to 10,000-12,000 MW and drops back within an
       hour): detected by `flag_implausible_readings`, then set to
       missing — every column of the hour, since the whole row is a
       scaled copy — so step 3 fills them like any other gap.
    3. **Missing hours** (spring-forward gaps, plus the real 2024-2025
       outage gaps, plus step 2's hours): filled by linear
       interpolation against a strictly regular hourly index. Unlike
       this project's original PJM version, the 2024-2025 gaps run up to
       83 consecutive hours — long enough that linear interpolation
       across the longest gaps is a much rougher approximation than it
       was for PJM's ~30 isolated missing hours. The same caveat applies
       to step 2's overnight stretches (a straight line cannot recreate
       the night-time dip). This is disclosed, not hidden:
       `gap_lengths()` and `implausible_readings()` report exactly what
       was filled.
    """
    grouped = _regular_hourly(df)
    grouped.loc[flag_implausible_readings(grouped)] = np.nan
    return grouped.interpolate(method="linear", limit_direction="both")


def load_clean_frame() -> pd.DataFrame:
    """Load and clean into a strictly-regular hourly frame indexed by
    timestamp, with `config.TARGET_COL` plus every exogenous column.
    See `clean_frame` for the cleaning logic.
    """
    return clean_frame(load_raw())


def gap_lengths(df: pd.DataFrame) -> pd.Series:
    """Length (in hours) of every consecutive run of missing hours in
    the raw file, against a strictly regular hourly index — used to
    show the real 2024-2025 outage gaps are structurally different from
    the 1-hour-per-year DST gaps, not just "a bit worse."
    """
    full_index = pd.date_range(df[config.DATETIME_COL].min(), df[config.DATETIME_COL].max(), freq="h")
    missing = pd.Series(full_index.difference(df[config.DATETIME_COL])).sort_values().reset_index(drop=True)
    if missing.empty:
        return pd.Series(dtype=int)
    is_new_run = (missing.diff().dt.total_seconds() != 3600).cumsum()
    return missing.groupby(is_new_run).size()
