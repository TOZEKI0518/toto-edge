from pathlib import Path
import csv
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path("toto-training-dataset.csv")
OUTPUT_PATH = Path("ml/feature_rfe.csv")
SELECTED_OUTPUT_PATH = Path("ml/selected_features.csv")

from common.features import FEATURE_COLUMNS

MIN_TRAIN_ROUNDS = 40
TEST_ROUND_WINDOW = 10
MIN_FEATURES = 12


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


def find_best_removal(df: pd.DataFrame, current_features: list[str]):
    baseline_accuracy = rolling_accuracy(df, current_features)

    best_feature_to_remove = None
    best_accuracy = baseline_accuracy

    for feature in current_features:
        test_features = [item for item in current_features if item != feature]
        accuracy = rolling_accuracy(df, test_features)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_feature_to_remove = feature

    return baseline_accuracy, best_feature_to_remove, best_accuracy


def save_selected_features(features: list[str], best_accuracy: float):
    SELECTED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SELECTED_OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["feature", "keep", "best_accuracy"],
        )
        writer.writeheader()

        for feature in FEATURE_COLUMNS:
            writer.writerow(
                {
                    "feature": feature,
                    "keep": feature in features,
                    "best_accuracy": best_accuracy,
                }
            )


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["result"]).copy()
    df = df.sort_values("roundNo").reset_index(drop=True)

    missing = [feature for feature in FEATURE_COLUMNS if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    current_features = FEATURE_COLUMNS.copy()

    print("===================================")
    print("Recursive Feature Elimination")
    print("===================================")
    print(f"Rows: {len(df)}")
    print(f"Start Features: {len(current_features)}")
    print(f"Minimum Features: {MIN_FEATURES}")
    print()

    history = []

    best_features = current_features.copy()
    best_accuracy = rolling_accuracy(df, current_features)

    print(f"Initial Accuracy: {best_accuracy:.4f}")
    print()

    step = 0

    while len(current_features) > MIN_FEATURES:
        step += 1

        baseline_accuracy, remove_feature, next_accuracy = find_best_removal(
            df,
            current_features,
        )

        if remove_feature is None:
            print("No further improvement. Stop.")
            break

        delta = next_accuracy - baseline_accuracy

        current_features = [
            feature for feature in current_features if feature != remove_feature
        ]

        if next_accuracy > best_accuracy:
            best_accuracy = next_accuracy
            best_features = current_features.copy()

        row = {
            "step": step,
            "feature_count_before": len(current_features) + 1,
            "removed_feature": remove_feature,
            "accuracy_before": baseline_accuracy,
            "accuracy_after": next_accuracy,
            "delta": delta,
            "remaining_features": len(current_features),
            "is_best": next_accuracy == best_accuracy,
        }

        history.append(row)

        print(
            f"Step {step:02d}: "
            f"Remove {remove_feature:<35} "
            f"{baseline_accuracy:.4f} -> {next_accuracy:.4f} "
            f"Delta {delta:+.4f} "
            f"Remaining {len(current_features)}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "feature_count_before",
                "removed_feature",
                "accuracy_before",
                "accuracy_after",
                "delta",
                "remaining_features",
                "is_best",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    save_selected_features(best_features, best_accuracy)

    print()
    print("===================================")
    print("RFE Summary")
    print("===================================")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Feature Count: {len(best_features)}")
    print()
    print("Selected Features:")
    for feature in best_features:
        print(f"- {feature}")

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {SELECTED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()