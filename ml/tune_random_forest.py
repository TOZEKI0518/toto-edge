from pathlib import Path
import csv
import itertools
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from common.dataset import load_dataset
from common.features import FEATURE_COLUMNS

OUTPUT_PATH = Path("ml/rf_tuning_results.csv")

MIN_TRAIN_ROUNDS = 40
TEST_ROUND_WINDOW = 10


PARAM_GRID = {
    "n_estimators": [500, 700, 900],
    "max_depth": [6, 7, 8, 9],
    "min_samples_leaf": [3, 4, 5, 6],
    "max_features": ["sqrt", "log2", None],
}


def create_model(params):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


def rolling_accuracy(df: pd.DataFrame, params: dict) -> float:
    rounds = sorted(df["roundNo"].unique())

    encoder = LabelEncoder()
    encoder.fit(df["result"])

    all_y_test = []
    all_preds = []

    start = MIN_TRAIN_ROUNDS

    while start < len(rounds):
        train_rounds = rounds[:start]
        test_rounds = rounds[start : start + TEST_ROUND_WINDOW]

        if not test_rounds:
            break

        train_df = df[df["roundNo"].isin(train_rounds)].copy()
        test_df = df[df["roundNo"].isin(test_rounds)].copy()

        X_train = train_df[FEATURE_COLUMNS].fillna(0)
        X_test = test_df[FEATURE_COLUMNS].fillna(0)

        y_train = encoder.transform(train_df["result"])
        y_test = encoder.transform(test_df["result"])

        model = create_model(params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        all_y_test.extend(y_test)
        all_preds.extend(preds)

        start += TEST_ROUND_WINDOW

    return accuracy_score(all_y_test, all_preds)


def main():
    df = load_dataset()
    df = df.sort_values("roundNo").reset_index(drop=True)

    missing = [feature for feature in FEATURE_COLUMNS if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[key] for key in keys]))

    print("===================================")
    print("RandomForest Hyperparameter Tuning")
    print("===================================")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Combinations: {len(combos)}")
    print()

    results = []

    for i, values in enumerate(combos, start=1):
        params = dict(zip(keys, values))
        accuracy = rolling_accuracy(df, params)

        row = {
            **params,
            "accuracy": accuracy,
        }
        results.append(row)

        print(
            f"{i:03d}/{len(combos)} "
            f"accuracy={accuracy:.4f} "
            f"params={params}"
        )

    results = sorted(results, key=lambda row: row["accuracy"], reverse=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "accuracy",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print("===================================")
    print("Best Results")
    print("===================================")

    for row in results[:10]:
        print(row)

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()