"""Train the production pipeline on the full dataset and save it.

Equivalent to the notebook's modeling steps, without the notebook.

Default is `LightGBM` — verified directly (see the model sweep in the
notebook/report) that it beats Linear Regression, Random Forest, and a
neural network (`MLPRegressor`) on every metric (MAE, RMSE, MAPE) in a
chronological holdout. It does *not* clearly beat the simplest
1-hour-ahead baseline ("same as the previous hour"); its advantage is
in multi-hour forecasting, where that baseline isn't available — see
the report's 24-hour comparison.

Uses the full dataset when `USE_FULL_KAGGLE_DATA=true` (how the
shipped model is trained), otherwise the bundled one-year sample.
"""

import argparse

from energy_demand_forecasting import config, data, features, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(model.MODEL_FACTORIES.keys()),
        default="LightGBM",
        help="Which model family to train (default: LightGBM).",
    )
    args = parser.parse_args()

    frame = data.load_clean_frame()
    X, y = features.split_features_target(frame)
    estimator = model.MODEL_FACTORIES[args.model]()
    pipeline = model.train_pipeline(X, y, estimator=estimator)
    model.save_pipeline(pipeline)
    print(f"Trained {args.model} on {len(X)} hourly rows. Saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
