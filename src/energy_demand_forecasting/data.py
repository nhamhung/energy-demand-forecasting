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

from pathlib import Path

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


def load_raw() -> pd.DataFrame:
    """Load the CSV exactly as published, only parsing the datetime
    column — no cleaning yet. Useful for demonstrating the DST/outage
    issues directly against the untouched data (see `config.py`'s
    docstring).
    """
    df = pd.read_csv(_require_file(config.RAW_CSV))
    df[config.DATETIME_COL] = pd.to_datetime(df[config.DATETIME_COL])
    return df.sort_values(config.DATETIME_COL).reset_index(drop=True)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the data-quality issues found by checking the raw file
    directly (see `config.py`'s module docstring):

    1. **Duplicate fall-back timestamps**: averaged into one reading
       per timestamp, across every numeric column — there's no way to
       tell which of the readings is "more correct" from this file
       alone, so averaging is the least-assuming choice.
    2. **Missing hours** (spring-forward gaps, plus the real 2024-2025
       outage gaps): filled by linear interpolation against a strictly
       regular hourly index. Unlike this project's original PJM
       version, the 2024-2025 gaps run up to 83 consecutive hours —
       long enough that linear interpolation across the longest gaps is
       a much rougher approximation than it was for PJM's ~30 isolated
       missing hours. This is disclosed, not hidden: `gap_lengths()`
       reports the actual distribution so it's checkable.
    """
    numeric_cols = [config.TARGET_COL] + config.EXOGENOUS_COLS
    grouped = df.groupby(config.DATETIME_COL)[numeric_cols].mean()
    full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq="h")
    grouped = grouped.reindex(full_index)
    grouped = grouped.interpolate(method="linear")
    grouped.index.name = config.DATETIME_COL
    return grouped


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
