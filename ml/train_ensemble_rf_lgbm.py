from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from common.dataset import load_dataset
from common.features import FEATURE_COLUMNS
from common.models import create_random_forest

MODEL_PATH = Path("ml/toto_ensemble_rf_lgbm.joblib")

MIN_TRAIN_ROUNDS = 40
TEST_ROUND_WINDOW = 10


def create_lgbm():
    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def align_probs(model, probs, target_classes):
    model_classes = list(model.classes_)
    aligned = np.zeros((probs.shape[0], len(target_classes)))

    for i, cls in enumerate(target_classes):
        if cls in model_classes:
            aligned[:, i] = probs[:, model_classes.index(cls)]

    return aligned


def evaluate_window(df, train_rounds, test_rounds, encoder):
    train_df = df[df["roundNo"].isin(train_rounds)].copy()
    test_df = df[df["roundNo"].isin(test_rounds)].copy()

    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    X_test = test_df[FEATURE_COLUMNS].fillna(0)

    y_train_label = train_df["result"]
    y_train_encoded = encoder.transform(train_df["result"])
    y_test = encoder.transform(test_df["result"])

    rf = create_random_forest()
    lgbm = create_lgbm()

    rf.fit(X_train, y_train_label)
    lgbm.fit(X_train, y_train_encoded)

    rf_probs = align_probs(rf, rf.predict_proba(X_test), encoder.classes_)
    lgbm_probs = lgbm.predict_proba(X_test)

    ensemble_probs = (rf_probs * 0.7) + (lgbm_probs * 0.3)
    preds = ensemble_probs.argmax(axis=1)

    acc = accuracy_score(y_test, preds)

    return {
        "rf": rf,
        "lgbm": lgbm,
        "y_test": y_test,
        "preds": preds,
        "accuracy": acc,
        "test_rows": len(test_df),
        "train_rows": len(train_df),
        "train_start": train_rounds[0],
        "train_end": train_rounds[-1],
        "test_start": test_rounds[0],
        "test_end": test_rounds[-1],
    }


def main():
    df = load_dataset()
    df = df.sort_values("roundNo").reset_index(drop=True)

    missing = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rounds = sorted(df["roundNo"].unique())

    encoder = LabelEncoder()
    encoder.fit(df["result"])

    results = []
    all_y = []
    all_preds = []

    start = MIN_TRAIN_ROUNDS

    print("===================================")
    print("RF + LightGBM Ensemble Rolling")
    print("===================================")
    print(f"Rows: {len(df)}")
    print(f"Rounds: {rounds[0]} - {rounds[-1]}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print()

    window = 1

    while start < len(rounds):
        train_rounds = rounds[:start]
        test_rounds = rounds[start : start + TEST_ROUND_WINDOW]

        if not test_rounds:
            break

        result = evaluate_window(df, train_rounds, test_rounds, encoder)
        results.append(result)

        all_y.extend(result["y_test"])
        all_preds.extend(result["preds"])

        print(
            f"Window {window}: "
            f"Train {result['train_start']}-{result['train_end']} "
            f"({result['train_rows']} rows) -> "
            f"Test {result['test_start']}-{result['test_end']} "
            f"({result['test_rows']} rows) | "
            f"Accuracy {result['accuracy']:.4f}"
        )

        start += TEST_ROUND_WINDOW
        window += 1

    accuracies = [r["accuracy"] for r in results]
    overall = accuracy_score(all_y, all_preds)

    print()
    print("===================================")
    print("Summary")
    print("===================================")
    print(f"Mean Accuracy: {sum(accuracies) / len(accuracies):.4f}")
    print(f"Best Accuracy: {max(accuracies):.4f}")
    print(f"Worst Accuracy: {min(accuracies):.4f}")
    print(f"Overall Accuracy: {overall:.4f}")
    print()

    print("Classification Report")
    print(classification_report(all_y, all_preds, target_names=encoder.classes_))

    print("Confusion Matrix")
    print(confusion_matrix(all_y, all_preds))

    final = results[-1]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "rf": final["rf"],
            "lgbm": final["lgbm"],
            "encoder": encoder,
            "features": FEATURE_COLUMNS,
            "rf_weight": 0.7,
            "lgbm_weight": 0.3,
            "overall_accuracy": overall,
        },
        MODEL_PATH,
    )

    print()
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()