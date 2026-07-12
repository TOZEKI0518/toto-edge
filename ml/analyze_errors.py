from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path("toto-training-dataset.csv")
OUTPUT_PATH = Path("ml/error_analysis.csv")

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


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["result"]).copy()
    df = df.sort_values("roundNo").reset_index(drop=True)

    rounds = sorted(df["roundNo"].unique())

    encoder = LabelEncoder()
    encoder.fit(df["result"])

    rows = []
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

        model = create_model()
        model.fit(X_train, y_train)

        pred_encoded = model.predict(X_test)
        pred_labels = encoder.inverse_transform(pred_encoded)

        probabilities = model.predict_proba(X_test)
        classes = list(encoder.classes_)

        for idx, (_, row) in enumerate(test_df.iterrows()):
            probability_by_class = {
                label: probabilities[idx][classes.index(label)]
                if label in classes
                else 0
                for label in ["H", "D", "A"]
            }

            predicted = pred_labels[idx]
            actual = row["result"]

            output = {
                "windowNo": len(rows) + 1,
                "roundNo": row["roundNo"],
                "snapshotDate": row["snapshotDate"],
                "league": row["league"],
                "homeTeam": row["homeTeam"],
                "awayTeam": row["awayTeam"],
                "prediction": predicted,
                "actual": actual,
                "correct": predicted == actual,
                "confidence": max(probability_by_class.values()),
                "probHome": probability_by_class["H"],
                "probDraw": probability_by_class["D"],
                "probAway": probability_by_class["A"],
            }

            for feature in FEATURE_COLUMNS:
                output[feature] = row.get(feature, 0)

            rows.append(output)

        start += TEST_ROUND_WINDOW

    result_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("===================================")
    print("Error Analysis Export")
    print("===================================")
    print(f"Rows: {len(result_df)}")
    print(f"Saved: {OUTPUT_PATH}")

    if len(result_df) == 0:
        return

    print()
    print("Overall Accuracy")
    print(result_df["correct"].mean())

    print()
    print("Accuracy by Round")
    print(result_df.groupby("roundNo")["correct"].mean().to_string())

    print()
    print("Accuracy by Actual Result")
    print(result_df.groupby("actual")["correct"].mean().to_string())

    print()
    print("Accuracy by Prediction")
    print(result_df.groupby("prediction")["correct"].mean().to_string())

    print()
    print("Worst Rounds")
    print(
        result_df.groupby("roundNo")["correct"]
        .agg(["count", "sum", "mean"])
        .sort_values("mean")
        .head(10)
        .to_string()
    )


if __name__ == "__main__":
    main()