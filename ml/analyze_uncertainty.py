from pathlib import Path
import pandas as pd
import joblib

from common.dataset import load_dataset
from common.features import FEATURE_COLUMNS
from common.models import create_random_forest

OUTPUT_PATH = Path("ml/uncertainty_analysis.csv")

MIN_TRAIN_ROUNDS = 40
TEST_ROUND_WINDOW = 10


def main():
    df = load_dataset()
    rounds = sorted(df["roundNo"].unique())

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

        y_train = train_df["result"]

        model = create_random_forest()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        classes = list(model.classes_)

        for i, (_, row) in enumerate(test_df.iterrows()):
            prob_h = probs[i][classes.index("H")] if "H" in classes else 0
            prob_d = probs[i][classes.index("D")] if "D" in classes else 0
            prob_a = probs[i][classes.index("A")] if "A" in classes else 0

            sorted_probs = sorted([prob_h, prob_d, prob_a], reverse=True)
            confidence = sorted_probs[0]
            margin = sorted_probs[0] - sorted_probs[1]
            entropy = -(sum(p * __import__("math").log(p + 1e-12) for p in [prob_h, prob_d, prob_a]))

            rows.append({
                "roundNo": row["roundNo"],
                "snapshotDate": row["snapshotDate"],
                "homeTeam": row["homeTeam"],
                "awayTeam": row["awayTeam"],
                "prediction": preds[i],
                "actual": row["result"],
                "correct": preds[i] == row["result"],
                "probH": prob_h,
                "probD": prob_d,
                "probA": prob_a,
                "confidence": confidence,
                "margin": margin,
                "entropy": entropy,
                "isDraw": row["result"] == "D",
            })

        start += TEST_ROUND_WINDOW

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("===================================")
    print("Uncertainty Analysis")
    print("===================================")
    print(f"Rows: {len(out)}")
    print(f"Saved: {OUTPUT_PATH}")
    print()

    print("Draw rate by confidence bucket")
    out["confidenceBucket"] = pd.cut(
        out["confidence"],
        bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0],
        include_lowest=True,
    )
    print(out.groupby("confidenceBucket", observed=False)["isDraw"].mean().to_string())

    print()
    print("Draw rate by margin bucket")
    out["marginBucket"] = pd.cut(
        out["margin"],
        bins=[0, 0.05, 0.10, 0.20, 0.40, 1.0],
        include_lowest=True,
    )
    print(out.groupby("marginBucket", observed=False)["isDraw"].mean().to_string())

    print()
    print("Top uncertain matches")
    print(
        out.sort_values(["margin", "confidence"])
        .head(20)[
            [
                "roundNo",
                "homeTeam",
                "awayTeam",
                "prediction",
                "actual",
                "probH",
                "probD",
                "probA",
                "confidence",
                "margin",
                "entropy",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()