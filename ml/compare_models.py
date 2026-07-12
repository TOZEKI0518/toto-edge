from __future__ import annotations

from pathlib import Path
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from common.dataset import load_dataset
from common.features import FEATURE_COLUMNS
from common.models import create_random_forest


OUTPUT_SUMMARY = Path("ml/model_comparison.csv")
OUTPUT_WINDOWS = Path("ml/model_comparison_windows.csv")
OUTPUT_PREDICTIONS = Path("ml/model_comparison_predictions.csv")

TARGET_CLASSES = ["A", "D", "H"]


def create_lightgbm():
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


def align_probabilities(
    model_classes,
    probabilities: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros(
        (probabilities.shape[0], len(TARGET_CLASSES)),
        dtype=float,
    )

    model_classes = list(model_classes)

    for target_index, target_class in enumerate(TARGET_CLASSES):
        if target_class in model_classes:
            source_index = model_classes.index(target_class)
            aligned[:, target_index] = probabilities[:, source_index]

    row_sums = aligned.sum(axis=1, keepdims=True)

    # 念のため、確率合計が0になる異常行を防ぐ
    zero_rows = row_sums[:, 0] == 0
    if zero_rows.any():
        aligned[zero_rows] = 1.0 / len(TARGET_CLASSES)
        row_sums = aligned.sum(axis=1, keepdims=True)

    return aligned / row_sums


def predictions_from_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    indexes = probabilities.argmax(axis=1)

    return np.array(
        [TARGET_CLASSES[index] for index in indexes],
        dtype=object,
    )


def calculate_metrics(
    actual: list[str],
    predicted: list[str],
    probabilities: np.ndarray,
) -> dict:
    accuracy = accuracy_score(actual, predicted)

    macro_f1 = f1_score(
        actual,
        predicted,
        labels=TARGET_CLASSES,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        actual,
        predicted,
        labels=TARGET_CLASSES,
        average="weighted",
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=TARGET_CLASSES,
        zero_division=0,
    )

    class_metrics = {
        target_class: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, target_class in enumerate(TARGET_CLASSES)
    }

    loss = log_loss(
        actual,
        probabilities,
        labels=TARGET_CLASSES,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "log_loss": float(loss),
        "draw_precision": class_metrics["D"]["precision"],
        "draw_recall": class_metrics["D"]["recall"],
        "draw_f1": class_metrics["D"]["f1"],
        "away_recall": class_metrics["A"]["recall"],
        "home_recall": class_metrics["H"]["recall"],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-train-rounds",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--test-window",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--rf-weight",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--lgbm-weight",
        type=float,
        default=0.3,
    )

    args = parser.parse_args()

    if args.min_train_rounds < 1:
        raise ValueError("--min-train-rounds must be at least 1.")

    if args.test_window < 1:
        raise ValueError("--test-window must be at least 1.")

    if args.rf_weight < 0 or args.lgbm_weight < 0:
        raise ValueError("Model weights cannot be negative.")

    weight_total = args.rf_weight + args.lgbm_weight

    if weight_total <= 0:
        raise ValueError("At least one model weight must be positive.")

    rf_weight = args.rf_weight / weight_total
    lgbm_weight = args.lgbm_weight / weight_total

    df = load_dataset()
    df = df[df["result"].isin(TARGET_CLASSES)].copy()
    df = df.sort_values(["roundNo"]).reset_index(drop=True)

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    rounds = sorted(df["roundNo"].unique())

    if len(rounds) <= args.min_train_rounds:
        raise ValueError(
            "Not enough rounds for the requested rolling backtest."
        )

    model_names = ["RandomForest", "LightGBM", "Ensemble"]

    all_actual = []
    all_probabilities = {
        model_name: []
        for model_name in model_names
    }

    window_rows = []
    prediction_rows = []

    start = args.min_train_rounds
    window_number = 1

    print("=" * 72)
    print("Fair Model Comparison")
    print("=" * 72)
    print(f"Rows              : {len(df)}")
    print(f"Rounds            : {rounds[0]} - {rounds[-1]}")
    print(f"Features          : {len(FEATURE_COLUMNS)}")
    print(f"Minimum train     : {args.min_train_rounds} rounds")
    print(f"Test window       : {args.test_window} rounds")
    print(
        f"Ensemble weights  : "
        f"RF {rf_weight:.2f} / LightGBM {lgbm_weight:.2f}"
    )
    print()

    while start < len(rounds):
        train_rounds = rounds[:start]
        test_rounds = rounds[
            start : start + args.test_window
        ]

        if not test_rounds:
            break

        train_df = df[
            df["roundNo"].isin(train_rounds)
        ].copy()

        test_df = df[
            df["roundNo"].isin(test_rounds)
        ].copy()

        x_train = train_df[FEATURE_COLUMNS].fillna(0)
        x_test = test_df[FEATURE_COLUMNS].fillna(0)

        y_train = train_df["result"].astype(str)
        y_test = test_df["result"].astype(str).tolist()

        rf = create_random_forest()
        lgbm = create_lightgbm()

        rf.fit(x_train, y_train)
        lgbm.fit(x_train, y_train)

        rf_probs = align_probabilities(
            rf.classes_,
            rf.predict_proba(x_test),
        )

        lgbm_probs = align_probabilities(
            lgbm.classes_,
            lgbm.predict_proba(x_test),
        )

        ensemble_probs = (
            rf_probs * rf_weight
            + lgbm_probs * lgbm_weight
        )

        probability_map = {
            "RandomForest": rf_probs,
            "LightGBM": lgbm_probs,
            "Ensemble": ensemble_probs,
        }

        print(
            f"Window {window_number}: "
            f"Train {train_rounds[0]}-{train_rounds[-1]} "
            f"({len(train_df)} rows) -> "
            f"Test {test_rounds[0]}-{test_rounds[-1]} "
            f"({len(test_df)} rows)"
        )

        for model_name, probabilities in probability_map.items():
            predictions = predictions_from_probabilities(
                probabilities
            )

            metrics = calculate_metrics(
                actual=y_test,
                predicted=predictions.tolist(),
                probabilities=probabilities,
            )

            window_rows.append(
                {
                    "window": window_number,
                    "model": model_name,
                    "train_round_start": train_rounds[0],
                    "train_round_end": train_rounds[-1],
                    "test_round_start": test_rounds[0],
                    "test_round_end": test_rounds[-1],
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    **metrics,
                }
            )

            all_probabilities[model_name].append(
                probabilities
            )

            print(
                f"  {model_name:<14} "
                f"Accuracy={metrics['accuracy']:.4f} "
                f"MacroF1={metrics['macro_f1']:.4f} "
                f"DrawRecall={metrics['draw_recall']:.4f} "
                f"LogLoss={metrics['log_loss']:.4f}"
            )

            for position, (_, row) in enumerate(
                test_df.iterrows()
            ):
                prediction_rows.append(
                    {
                        "window": window_number,
                        "roundNo": row["roundNo"],
                        "homeTeam": row.get("homeTeam", ""),
                        "awayTeam": row.get("awayTeam", ""),
                        "model": model_name,
                        "actual": y_test[position],
                        "prediction": predictions[position],
                        "correct": (
                            predictions[position]
                            == y_test[position]
                        ),
                        "probA": probabilities[position, 0],
                        "probD": probabilities[position, 1],
                        "probH": probabilities[position, 2],
                    }
                )

        all_actual.extend(y_test)

        print()

        start += args.test_window
        window_number += 1

    summary_rows = []

    print("=" * 72)
    print("Overall Results")
    print("=" * 72)

    for model_name in model_names:
        probabilities = np.vstack(
            all_probabilities[model_name]
        )

        predictions = predictions_from_probabilities(
            probabilities
        )

        metrics = calculate_metrics(
            actual=all_actual,
            predicted=predictions.tolist(),
            probabilities=probabilities,
        )

        summary_rows.append(
            {
                "model": model_name,
                "rows": len(all_actual),
                "features": len(FEATURE_COLUMNS),
                "rf_weight": (
                    rf_weight
                    if model_name == "Ensemble"
                    else ""
                ),
                "lgbm_weight": (
                    lgbm_weight
                    if model_name == "Ensemble"
                    else ""
                ),
                **metrics,
            }
        )

        print()
        print(model_name)
        print("-" * 40)
        print(f"Accuracy       : {metrics['accuracy']:.4f}")
        print(f"Macro F1       : {metrics['macro_f1']:.4f}")
        print(f"Weighted F1    : {metrics['weighted_f1']:.4f}")
        print(f"Log Loss       : {metrics['log_loss']:.4f}")
        print(f"Draw Precision : {metrics['draw_precision']:.4f}")
        print(f"Draw Recall    : {metrics['draw_recall']:.4f}")
        print(f"Draw F1        : {metrics['draw_f1']:.4f}")
        print(f"Away Recall    : {metrics['away_recall']:.4f}")
        print(f"Home Recall    : {metrics['home_recall']:.4f}")

        print()
        print("Classification Report")
        print(
            classification_report(
                all_actual,
                predictions,
                labels=TARGET_CLASSES,
                zero_division=0,
            )
        )

        print("Confusion Matrix [A, D, H]")
        print(
            confusion_matrix(
                all_actual,
                predictions,
                labels=TARGET_CLASSES,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    window_df = pd.DataFrame(window_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    summary_df = summary_df.sort_values(
        ["accuracy", "macro_f1", "log_loss"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    OUTPUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    window_df.to_csv(
        OUTPUT_WINDOWS,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
        encoding="utf-8-sig",
    )

    winner = summary_df.iloc[0]

    print()
    print("=" * 72)
    print("Comparison Ranking")
    print("=" * 72)

    print(
        summary_df[
            [
                "model",
                "accuracy",
                "macro_f1",
                "log_loss",
                "draw_precision",
                "draw_recall",
                "draw_f1",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Accuracy Winner: {winner['model']}")
    print(f"Accuracy       : {winner['accuracy']:.4f}")
    print()
    print(f"Saved: {OUTPUT_SUMMARY}")
    print(f"Saved: {OUTPUT_WINDOWS}")
    print(f"Saved: {OUTPUT_PREDICTIONS}")


if __name__ == "__main__":
    main()