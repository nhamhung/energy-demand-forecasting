"""Train the production pipeline on the full dataset and save it.

Equivalent to the notebook's modeling steps, without the notebook.

Default is `LightGBM` — verified directly (see the model sweep in the
notebook/report) that it beats Linear Regression, Random Forest, and a
neural network (`MLPRegressor`) on every metric (MAE, RMSE, MAPE) in a
chronological holdout.
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
