from pathlib import Path
import pandas as pd
import xgboost as xgb

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_PATH = Path("toto-training-dataset.csv")
MODEL_PATH = Path("ml/toto_xgboost.joblib")

from common.features import FEATURE_COLUMNS

MIN_TRAIN_ROUNDS = 40
TEST_ROUND_WINDOW = 10


def create_model():
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss",
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

    model = create_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    return {
        "model": model,
        "train_round_start": train_rounds[0],
        "train_round_end": train_rounds[-1],
        "test_round_start": test_rounds[0],
        "test_round_end": test_rounds[-1],
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "accuracy": accuracy,
        "y_test": y_test,
        "preds": preds,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["result"]).copy()
    df = df.sort_values("roundNo").reset_index(drop=True)

    missing = [feature for feature in FEATURE_COLUMNS if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

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
    print("XGBoost Rolling Time Backtest")
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
            f"({result['train_rows']} rows) -> "
            f"Test {result['test_round_start']}-{result['test_round_end']} "
            f"({result['test_rows']} rows) | "
            f"Accuracy {result['accuracy']:.4f}"
        )

    accuracies = [result["accuracy"] for result in results]

    all_y_test = []
    all_preds = []

    for result in results:
        all_y_test.extend(result["y_test"])
        all_preds.extend(result["preds"])

    overall_accuracy = accuracy_score(all_y_test, all_preds)

    print()
    print("===================================")
    print("Rolling Summary")
    print("===================================")
    print(f"Mean Accuracy: {sum(accuracies) / len(accuracies):.4f}")
    print(f"Best Accuracy: {max(accuracies):.4f}")
    print(f"Worst Accuracy: {min(accuracies):.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print()

    print("Classification Report")
    print(classification_report(all_y_test, all_preds, target_names=encoder.classes_))

    print("Confusion Matrix")
    print(confusion_matrix(all_y_test, all_preds))

    final_train_rounds = rounds[:-TEST_ROUND_WINDOW]
    final_test_rounds = rounds[-TEST_ROUND_WINDOW:]

    final_result = evaluate_window(df, final_train_rounds, final_test_rounds, encoder)

    if final_result:
        final_model = final_result["model"]

        importance = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": final_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print()
        print("Feature Importance")
        print(importance.to_string(index=False))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": final_model,
                "encoder": encoder,
                "features": FEATURE_COLUMNS,
                "rolling_mean_accuracy": float(sum(accuracies) / len(accuracies)),
                "rolling_overall_accuracy": float(overall_accuracy),
            },
            MODEL_PATH,
        )

        print()
        print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()