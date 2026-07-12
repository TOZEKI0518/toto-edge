from pathlib import Path
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from common.dataset import load_dataset
from common.features import FEATURE_COLUMNS

MODEL_PATH = Path("ml/toto_lightgbm_v2.joblib")

MIN_TRAIN_ROUNDS = 40
TEST_ROUND_WINDOW = 10


def create_model():
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


def evaluate_window(df, train_rounds, test_rounds, encoder):
    train_df = df[df["roundNo"].isin(train_rounds)].copy()
    test_df = df[df["roundNo"].isin(test_rounds)].copy()

    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    X_test = test_df[FEATURE_COLUMNS].fillna(0)

    y_train = encoder.transform(train_df["result"])
    y_test = encoder.transform(test_df["result"])

    model = create_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, y_test, preds, acc


def main():
    df = load_dataset()
    df = df.sort_values("roundNo").reset_index(drop=True)

    missing = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rounds = sorted(df["roundNo"].unique())

    encoder = LabelEncoder()
    encoder.fit(df["result"])

    all_y = []
    all_preds = []
    accuracies = []
    final_model = None

    start = MIN_TRAIN_ROUNDS

    print("===================================")
    print("LightGBM v2 Rolling Backtest")
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

        model, y_test, preds, acc = evaluate_window(
            df, train_rounds, test_rounds, encoder
        )

        final_model = model
        accuracies.append(acc)
        all_y.extend(y_test)
        all_preds.extend(preds)

        print(
            f"Window {window}: "
            f"Train {train_rounds[0]}-{train_rounds[-1]} "
            f"-> Test {test_rounds[0]}-{test_rounds[-1]} "
            f"| Accuracy {acc:.4f}"
        )

        start += TEST_ROUND_WINDOW
        window += 1

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

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print()
    print("Feature Importance")
    print(importance.to_string(index=False))

    joblib.dump(
        {
            "model": final_model,
            "encoder": encoder,
            "features": FEATURE_COLUMNS,
            "overall_accuracy": overall,
        },
        MODEL_PATH,
    )

    print()
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()