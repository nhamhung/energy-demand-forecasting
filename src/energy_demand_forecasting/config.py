"""Paths, constants, and column schema for the Romanian hourly
electricity consumption/production dataset.

Dataset: hourly electricity `Consumption` and `Production` (megawatts)
for Romania's national grid, split by generation source (`Nuclear`,
`Wind`, `Hydroelectric`, `Oil and Gas`, `Coal`, `Solar`, `Biomass`) —
`stefancomanita/hourly-electricity-consumption-and-production` on
Kaggle.

**This dataset is live and continuously updated** — checked directly:
a first download returned 46,002 rows through 2024-04-01; a second
download weeks later returned 62,810 rows through 2026-03-14, with the
same historical rows unchanged. That's unlike this portfolio's other
projects, all frozen historical snapshots. The specific row count and
date range quoted throughout this project reflect one snapshot
(downloaded 2026-09-30); re-downloading later will add more recent
rows without invalidating anything already documented here, since the
past data itself doesn't change.

Verified directly against the real downloaded CSV (not assumed):

- **Two genuine Daylight Saving Time issues**, the same underlying
  cause as any DST-observing country: a missing hour every spring (the
  "spring forward" hour never happens) and a duplicated hour every
  autumn (the "fall back" hour happens twice) — Romania's EEST->EET
  transition duplicates 3am local time, not 2am like the US, since
  Romania is in the Eastern European time zone.
- **A second, unrelated data-quality issue only in the most recent
  years**: real multi-hour gaps (up to 83 consecutive missing hours in
  February 2025) concentrated in 2024-2025, consistent with genuine
  reporting/scraping outages in whatever live feed this dataset's
  maintainer scrapes — not a DST artifact, and not present at all in
  2019-2023.
- **The generation-mix columns (`Production` and its 7 source
  breakdowns) are recorded at the same hour as `Consumption`, not
  before it** — using them directly as features to predict that same
  hour's `Consumption` would be a genuine leakage trap (a real
  forecasting system doesn't know this hour's actual generation mix
  before the hour happens either). See `features.py` for how this
  project avoids it: only *lagged* versions of these columns are used.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_CSV = DATA_RAW_DIR / "Romania_hourly.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"

# --- Kaggle source ---------------------------------------------------------

KAGGLE_DATASET = "stefancomanita/hourly-electricity-consumption-and-production"

# --- Columns -------------------------------------------------------------

DATETIME_COL = "DateTime"
TARGET_COL = "Consumption"

# The generation-mix columns — recorded at the same hour as Consumption
# (see module docstring). Never used unlagged as features.
EXOGENOUS_COLS = ["Production", "Nuclear", "Wind", "Hydroelectric", "Oil and Gas", "Coal", "Solar", "Biomass"]
RENEWABLE_COLS = ["Wind", "Hydroelectric", "Solar"]

RANDOM_SEED = 42

# --- Modeling -------------------------------------------------------------
LAG_HOURS = [24, 168]
ROLLING_WINDOWS_HOURS = [24, 168]
EXOGENOUS_LAG_HOURS = 24  # only "yesterday, same hour" for the 8 exogenous columns — kept to one lag to avoid an explosion of near-duplicate features across 8 columns

TEST_FRACTION = 0.15

# Multi-step recursive forecasting is capped at 24 hours ahead, not
# days — matching a related piece of prior work on a different
# electricity dataset, which found multi-step forecasts hold up much
# better over a short (same-day) horizon than a genuinely long one. It
# also has a nice mechanical consequence: since EXOGENOUS_LAG_HOURS is
# also 24, every generation-mix feature stays grounded in real data for
# the *entire* forecast window (their 24h-ago reference point is always
# at or before the forecast's anchor hour) — only the rolling-window
# features (which need the immediately preceding hours) ever need to
# read the model's own just-made predictions. See `model.recursive_forecast`.
MAX_FORECAST_HORIZON_HOURS = 24
