from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_PATH = Path("toto-training-dataset.csv")
MODEL_PATH = Path("ml/toto_calibrated_random_forest.joblib")

FEATURE_COLUMNS = [
    "homeAttackRating",
    "defenseRatingDiff",
    "awayAttackRating",
    "homeAwayDiff",
    "homeAwayGoalsPerMatchDiff",
    "homeRank",
    "awayAwayGoalsAgainstPerMatch",
    "awayPoints",
    "awayAwayGoalsPerMatch",
    "pointsDiff",
    "rankDiff",
    "homeHomeWinRate",
    "homeHomeElo",
    "homeHomeGoalsAgainstPerMatch",
    "homeHomeGoalsPerMatch",
    "absEloDiff",
    "absRankDiff",
    "absPointsDiff",
    "absAttackDiff",
    "absDefenseDiff",
    "balanceScore",
]

MIN_TRAIN_ROUNDS = 40
TEST_ROUND_WINDOW = 10


def create_base_model():
    return RandomForestClassifier(
        n_estimators=700,
        max_depth=7,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
    )


def evaluate_window(df, train_rounds, test_rounds, encoder):
    train_df = df[df["roundNo"].isin(train_rounds)].copy()
    test_df = df[df["roundNo"].isin(test_rounds)].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        return None

    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    X_test = test_df[FEATURE_COLUMNS].fillna(0)

    y_train = encoder.transform(train_df["result"])
    y_test = encoder.transform(test_df["result"])

    base_model = create_base_model()

    model = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=3,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    return {
        "model": model,
        "train_round_start": train_rounds[0],
        "train_round_end": train_rounds[-1],
        "test_round_start": test_rounds[0],
        "test_round_end": test_rounds[-1],
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "accuracy": accuracy_score(y_test, preds),
        "log_loss": log_loss(y_test, probs, labels=list(range(len(encoder.classes_)))),
        "y_test": y_test,
        "preds": preds,
        "probs": probs,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["result"]).copy()
    df = df.sort_values("roundNo").reset_index(drop=True)

    rounds = sorted(df["roundNo"].unique())

    encoder = LabelEncoder()
    encoder.fit(df["result"])

    results = []
    start = MIN_TRAIN_ROUNDS

    while start < len(rounds):
        train_rounds = rounds[:start]
        test_rounds = rounds[start : start + TEST_ROUND_WINDOW]

        if len(test_rounds) == 0:
            break

        result = evaluate_window(df, train_rounds, test_rounds, encoder)
        if result:
            results.append(result)

        start += TEST_ROUND_WINDOW

    print("===================================")
    print("Calibrated Random Forest Rolling")
    print("===================================")
    print(f"Rows: {len(df)}")
    print(f"Rounds: {rounds[0]} - {rounds[-1]}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Windows: {len(results)}")
    print()

    for index, result in enumerate(results, start=1):
        print(
            f"Window {index}: "
            f"Train {result['train_round_start']}-{result['train_round_end']} "
            f"-> Test {result['test_round_start']}-{result['test_round_end']} | "
            f"Accuracy {result['accuracy']:.4f} | "
            f"LogLoss {result['log_loss']:.4f}"
        )

    accuracies = [result["accuracy"] for result in results]
    losses = [result["log_loss"] for result in results]

    all_y_test = []
    all_preds = []
    all_probs = []

    for result in results:
        all_y_test.extend(result["y_test"])
        all_preds.extend(result["preds"])
        all_probs.extend(result["probs"])

    print()
    print("===================================")
    print("Rolling Summary")
    print("===================================")
    print(f"Mean Accuracy: {sum(accuracies) / len(accuracies):.4f}")
    print(f"Mean LogLoss: {sum(losses) / len(losses):.4f}")
    print(f"Overall Accuracy: {accuracy_score(all_y_test, all_preds):.4f}")
    print(
        f"Overall LogLoss: "
        f"{log_loss(all_y_test, all_probs, labels=list(range(len(encoder.classes_)))):.4f}"
    )
    print()

    print("Classification Report")
    print(classification_report(all_y_test, all_preds, target_names=encoder.classes_))

    print("Confusion Matrix")
    print(confusion_matrix(all_y_test, all_preds))

    final_train_rounds = rounds[:-TEST_ROUND_WINDOW]
    final_test_rounds = rounds[-TEST_ROUND_WINDOW:]
    final_result = evaluate_window(df, final_train_rounds, final_test_rounds, encoder)

    if final_result:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": final_result["model"],
                "encoder": encoder,
                "features": FEATURE_COLUMNS,
                "rolling_mean_accuracy": float(sum(accuracies) / len(accuracies)),
                "rolling_mean_log_loss": float(sum(losses) / len(losses)),
            },
            MODEL_PATH,
        )
        print()
        print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()