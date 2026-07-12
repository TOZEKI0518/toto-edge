from pathlib import Path
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_PATH = Path("toto-training-dataset.csv")
MODEL_PATH = Path("ml/toto_lightgbm.joblib")

FEATURE_COLUMNS = [
    "defenseRatingDiff",
    "rankDiff",
    "homeAwayGoalsPerMatchDiff",
    "homeRank",
    "homeHomeElo",
    "pointsDiff",
    "homeHomeGoalsPerMatch",
    "homeAwayDiff",
    "homeAttackRating",
    "homeHomeGoalsAgainstPerMatch",
    "awayAwayGoalsPerMatch",
    "awayAttackRating",
    "homeHomeWinRate",
    "awayPoints",
    "awayAwayGoalsAgainstPerMatch",
]

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["result"]).copy()

    X = df[FEATURE_COLUMNS].fillna(0)
    y = df["result"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded,
    )

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(encoder.classes_),
        n_estimators=120,
        learning_rate=0.04,
        max_depth=3,
        num_leaves=7,
        min_child_samples=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    print("===================================")
    print("LightGBM Training Result")
    print("===================================")
    print(f"Rows: {len(df)}")
    print(f"Train: {len(X_train)}")
    print(f"Test: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print()

    print("Classification Report")
    print(classification_report(y_test, preds, target_names=encoder.classes_))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, preds))

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print()
    print("Feature Importance")
    print(importance.to_string(index=False))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "features": FEATURE_COLUMNS,
            "accuracy": accuracy,
        },
        MODEL_PATH,
    )

    print()
    print(f"Saved model: {MODEL_PATH}")

if __name__ == "__main__":
    main()