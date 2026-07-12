from pathlib import Path
import csv
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path("toto-training-dataset.csv")
OUTPUT_PATH = Path("ml/feature_ablation.csv")

from common.features import FEATURE_COLUMNS

MIN_TRAIN_ROUNDS = 40
TEST_ROUND_WINDOW = 10


def create_model():
    return RandomForestClassifier(
        n_estimators=700,
        max_depth=7,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
    )


def rolling_accuracy(df: pd.DataFrame, features: list[str]) -> float:
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

        if len(train_df) == 0 or len(test_df) == 0:
            start += TEST_ROUND_WINDOW
            continue

        X_train = train_df[features].fillna(0)
        X_test = test_df[features].fillna(0)

        y_train = encoder.transform(train_df["result"])
        y_test = encoder.transform(test_df["result"])

        model = create_model()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        all_y_test.extend(y_test)
        all_preds.extend(preds)

        start += TEST_ROUND_WINDOW

    if not all_y_test:
        return 0.0

    return accuracy_score(all_y_test, all_preds)


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["result"]).copy()
    df = df.sort_values("roundNo").reset_index(drop=True)

    missing = [feature for feature in FEATURE_COLUMNS if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    print("===================================")
    print("Feature Ablation Study")
    print("===================================")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print()

    baseline_accuracy = rolling_accuracy(df, FEATURE_COLUMNS)

    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print()

    results = [
        {
            "removed_feature": "__BASELINE__",
            "feature_count": len(FEATURE_COLUMNS),
            "accuracy": baseline_accuracy,
            "delta": 0.0,
            "judgement": "baseline",
        }
    ]

    for feature in FEATURE_COLUMNS:
        test_features = [item for item in FEATURE_COLUMNS if item != feature]
        accuracy = rolling_accuracy(df, test_features)
        delta = accuracy - baseline_accuracy

        if delta <= -0.005:
            judgement = "important_keep"
        elif delta >= 0.005:
            judgement = "candidate_remove"
        else:
            judgement = "neutral"

        results.append(
            {
                "removed_feature": feature,
                "feature_count": len(test_features),
                "accuracy": accuracy,
                "delta": delta,
                "judgement": judgement,
            }
        )

        print(
            f"- {feature:<35} "
            f"Accuracy {accuracy:.4f} "
            f"Delta {delta:+.4f} "
            f"{judgement}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "removed_feature",
                "feature_count",
                "accuracy",
                "delta",
                "judgement",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print("===================================")
    print("Most Important Features")
    print("===================================")

    important = sorted(
        [row for row in results if row["removed_feature"] != "__BASELINE__"],
        key=lambda row: row["delta"],
    )[:10]

    for row in important:
        print(
            f"{row['removed_feature']:<35} "
            f"Delta {row['delta']:+.4f}"
        )

    print()
    print("===================================")
    print("Remove Candidates")
    print("===================================")

    removable = sorted(
        [row for row in results if row["removed_feature"] != "__BASELINE__"],
        key=lambda row: row["delta"],
        reverse=True,
    )[:10]

    for row in removable:
        print(
            f"{row['removed_feature']:<35} "
            f"Delta {row['delta']:+.4f}"
        )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()