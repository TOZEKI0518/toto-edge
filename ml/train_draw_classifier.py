from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

from common.dataset import load_dataset
from common.features import FEATURE_COLUMNS

MODEL_PATH = Path("ml/toto_draw_classifier.joblib")

MIN_TRAIN_ROUNDS = 40
TEST_ROUND_WINDOW = 10


def create_draw_model():
    return RandomForestClassifier(
        n_estimators=700,
        max_depth=7,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


def evaluate_window(df, train_rounds, test_rounds):
    train_df = df[df["roundNo"].isin(train_rounds)].copy()
    test_df = df[df["roundNo"].isin(test_rounds)].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        return None

    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    X_test = test_df[FEATURE_COLUMNS].fillna(0)

    y_train = (train_df["result"] == "D").astype(int)
    y_test = (test_df["result"] == "D").astype(int)

    model = create_draw_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "train_round_start": train_rounds[0],
        "train_round_end": train_rounds[-1],
        "test_round_start": test_rounds[0],
        "test_round_end": test_rounds[-1],
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "y_test": y_test.tolist(),
        "preds": preds.tolist(),
        "probs": probs.tolist(),
    }


def safe_roc_auc(y_true, probs):
    if len(set(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, probs)


def main():
    df = load_dataset()
    df = df.sort_values("roundNo").reset_index(drop=True)

    missing = [feature for feature in FEATURE_COLUMNS if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    rounds = sorted(df["roundNo"].unique())

    results = []
    start = MIN_TRAIN_ROUNDS

    while start < len(rounds):
        train_rounds = rounds[:start]
        test_rounds = rounds[start : start + TEST_ROUND_WINDOW]

        if len(test_rounds) == 0:
            break

        result = evaluate_window(df, train_rounds, test_rounds)
        if result:
            results.append(result)

        start += TEST_ROUND_WINDOW

    print("===================================")
    print("Draw Specialist Rolling Backtest")
    print("===================================")
    print(f"Rows: {len(df)}")
    print(f"Rounds: {rounds[0]} - {rounds[-1]}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Windows: {len(results)}")
    print(f"Draw Rate: {(df['result'] == 'D').mean():.4f}")
    print()

    all_y_test = []
    all_preds = []
    all_probs = []

    for index, result in enumerate(results, start=1):
        y_test = result["y_test"]
        preds = result["preds"]
        probs = result["probs"]

        recall = recall_score(y_test, preds, zero_division=0)
        precision = precision_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(
            f"Window {index}: "
            f"Train {result['train_round_start']}-{result['train_round_end']} "
            f"({result['train_rows']} rows) -> "
            f"Test {result['test_round_start']}-{result['test_round_end']} "
            f"({result['test_rows']} rows) | "
            f"Precision {precision:.4f} | "
            f"Recall {recall:.4f} | "
            f"F1 {f1:.4f}"
        )

        all_y_test.extend(y_test)
        all_preds.extend(preds)
        all_probs.extend(probs)

    print()
    print("===================================")
    print("Draw Specialist Summary")
    print("===================================")
    print(f"Accuracy: {accuracy_score(all_y_test, all_preds):.4f}")
    print(f"Precision: {precision_score(all_y_test, all_preds, zero_division=0):.4f}")
    print(f"Recall: {recall_score(all_y_test, all_preds, zero_division=0):.4f}")
    print(f"F1: {f1_score(all_y_test, all_preds, zero_division=0):.4f}")
    print(f"ROC AUC: {safe_roc_auc(all_y_test, all_probs):.4f}")
    print(f"PR AUC: {average_precision_score(all_y_test, all_probs):.4f}")
    print()

    print("Classification Report")
    print(
        classification_report(
            all_y_test,
            all_preds,
            target_names=["Not Draw", "Draw"],
            zero_division=0,
        )
    )

    print("Confusion Matrix")
    print(confusion_matrix(all_y_test, all_preds))

    final_train_rounds = rounds[:-TEST_ROUND_WINDOW]
    final_test_rounds = rounds[-TEST_ROUND_WINDOW:]
    final_result = evaluate_window(df, final_train_rounds, final_test_rounds)

    if final_result:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": final_result["model"],
                "features": FEATURE_COLUMNS,
                "task": "draw_classifier",
            },
            MODEL_PATH,
        )

        print()
        print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()