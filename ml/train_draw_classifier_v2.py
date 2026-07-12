from __future__ import annotations

from pathlib import Path
import argparse

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from common.dataset import load_dataset
from common.features import FEATURE_COLUMNS
from common.models import create_random_forest


TARGET_CLASSES = ["A", "D", "H"]

OUTPUT_ALPHA_RESULTS = Path("ml/draw_v2_alpha_results.csv")
OUTPUT_WINDOW_RESULTS = Path("ml/draw_v2_window_results.csv")
OUTPUT_PREDICTIONS = Path("ml/draw_v2_predictions.csv")
MODEL_PATH = Path("ml/toto_draw_classifier_v2.joblib")


def create_lightgbm_multiclass():
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


def create_draw_classifier():
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=7,
        min_child_samples=25,
        subsample=0.90,
        colsample_bytree=0.90,
        class_weight="balanced",
        reg_alpha=0.5,
        reg_lambda=1.0,
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

    zero_rows = row_sums[:, 0] <= 0
    if zero_rows.any():
        aligned[zero_rows] = 1.0 / len(TARGET_CLASSES)
        row_sums = aligned.sum(axis=1, keepdims=True)

    return aligned / row_sums


def prediction_labels(probabilities: np.ndarray) -> np.ndarray:
    indexes = probabilities.argmax(axis=1)

    return np.array(
        [TARGET_CLASSES[index] for index in indexes],
        dtype=object,
    )


def adjust_draw_probability(
    base_probabilities: np.ndarray,
    specialist_draw_probabilities: np.ndarray,
    alpha: float,
    minimum_draw_probability: float = 0.02,
    maximum_draw_probability: float = 0.65,
) -> np.ndarray:
    """
    EnsembleのDraw確率とDraw専門モデルの確率をブレンドする。

    Dを補正した後、残りの確率をA/Hへ元の比率で配分する。
    """
    adjusted = base_probabilities.copy()

    base_a = base_probabilities[:, 0]
    base_d = base_probabilities[:, 1]
    base_h = base_probabilities[:, 2]

    new_d = (
        (1.0 - alpha) * base_d
        + alpha * specialist_draw_probabilities
    )

    new_d = np.clip(
        new_d,
        minimum_draw_probability,
        maximum_draw_probability,
    )

    remaining = 1.0 - new_d
    non_draw_total = base_a + base_h

    safe_total = np.where(
        non_draw_total > 0,
        non_draw_total,
        1.0,
    )

    adjusted[:, 0] = remaining * (base_a / safe_total)
    adjusted[:, 1] = new_d
    adjusted[:, 2] = remaining * (base_h / safe_total)

    zero_non_draw = non_draw_total <= 0

    if zero_non_draw.any():
        adjusted[zero_non_draw, 0] = remaining[zero_non_draw] / 2
        adjusted[zero_non_draw, 2] = remaining[zero_non_draw] / 2

    row_sums = adjusted.sum(axis=1, keepdims=True)

    return adjusted / row_sums


def calculate_metrics(
    actual: list[str],
    probabilities: np.ndarray,
) -> dict:
    predicted = prediction_labels(probabilities)

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

    draw_index = TARGET_CLASSES.index("D")

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "log_loss": float(
            log_loss(
                actual,
                probabilities,
                labels=TARGET_CLASSES,
            )
        ),
        "draw_precision": float(precision[draw_index]),
        "draw_recall": float(recall[draw_index]),
        "draw_f1": float(f1[draw_index]),
        "draw_support": int(support[draw_index]),
        "predicted_draws": int((predicted == "D").sum()),
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

    parser.add_argument(
        "--alphas",
        default="0.00,0.10,0.20,0.30,0.40,0.50,0.60",
        help="Draw補正強度。カンマ区切り。",
    )

    args = parser.parse_args()

    alphas = [
        float(value.strip())
        for value in args.alphas.split(",")
        if value.strip()
    ]

    if not alphas:
        raise ValueError("At least one alpha is required.")

    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Each alpha must be between 0 and 1.")

    weight_total = args.rf_weight + args.lgbm_weight

    if weight_total <= 0:
        raise ValueError("Model weights must total more than zero.")

    rf_weight = args.rf_weight / weight_total
    lgbm_weight = args.lgbm_weight / weight_total

    df = load_dataset()
    df = df[df["result"].isin(TARGET_CLASSES)].copy()
    df = df.sort_values("roundNo").reset_index(drop=True)

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    rounds = sorted(df["roundNo"].unique())

    if len(rounds) <= args.min_train_rounds:
        raise ValueError("Not enough rounds for rolling evaluation.")

    all_actual: list[str] = []
    all_base_probabilities: list[np.ndarray] = []
    all_draw_probabilities: list[np.ndarray] = []

    window_rows = []
    prediction_rows = []

    start = args.min_train_rounds
    window_number = 1

    print("=" * 72)
    print("Draw Classifier v2 Rolling Comparison")
    print("=" * 72)
    print(f"Rows              : {len(df)}")
    print(f"Rounds            : {rounds[0]} - {rounds[-1]}")
    print(f"Features          : {len(FEATURE_COLUMNS)}")
    print(f"Minimum train     : {args.min_train_rounds} rounds")
    print(f"Test window       : {args.test_window} rounds")
    print(
        f"Base Ensemble     : "
        f"RF {rf_weight:.2f} / LightGBM {lgbm_weight:.2f}"
    )
    print(f"Alphas            : {alphas}")
    print()

    while start < len(rounds):
        train_rounds = rounds[:start]
        test_rounds = rounds[start : start + args.test_window]

        if not test_rounds:
            break

        train_df = df[df["roundNo"].isin(train_rounds)].copy()
        test_df = df[df["roundNo"].isin(test_rounds)].copy()

        x_train = train_df[FEATURE_COLUMNS].fillna(0)
        x_test = test_df[FEATURE_COLUMNS].fillna(0)

        y_train = train_df["result"].astype(str)
        y_test = test_df["result"].astype(str).tolist()

        y_train_draw = (y_train == "D").astype(int)

        rf = create_random_forest()
        lgbm_model = create_lightgbm_multiclass()
        draw_model = create_draw_classifier()

        rf.fit(x_train, y_train)
        lgbm_model.fit(x_train, y_train)
        draw_model.fit(x_train, y_train_draw)

        rf_probabilities = align_probabilities(
            rf.classes_,
            rf.predict_proba(x_test),
        )

        lgbm_probabilities = align_probabilities(
            lgbm_model.classes_,
            lgbm_model.predict_proba(x_test),
        )

        base_probabilities = (
            rf_probabilities * rf_weight
            + lgbm_probabilities * lgbm_weight
        )

        draw_probabilities = draw_model.predict_proba(x_test)[:, 1]

        all_actual.extend(y_test)
        all_base_probabilities.append(base_probabilities)
        all_draw_probabilities.append(draw_probabilities)

        print(
            f"Window {window_number}: "
            f"Train {train_rounds[0]}-{train_rounds[-1]} "
            f"({len(train_df)} rows) -> "
            f"Test {test_rounds[0]}-{test_rounds[-1]} "
            f"({len(test_df)} rows)"
        )

        for alpha in alphas:
            adjusted = adjust_draw_probability(
                base_probabilities=base_probabilities,
                specialist_draw_probabilities=draw_probabilities,
                alpha=alpha,
            )

            metrics = calculate_metrics(
                actual=y_test,
                probabilities=adjusted,
            )

            window_rows.append(
                {
                    "window": window_number,
                    "alpha": alpha,
                    "train_round_start": train_rounds[0],
                    "train_round_end": train_rounds[-1],
                    "test_round_start": test_rounds[0],
                    "test_round_end": test_rounds[-1],
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    **metrics,
                }
            )

            print(
                f"  alpha={alpha:.2f} "
                f"Accuracy={metrics['accuracy']:.4f} "
                f"MacroF1={metrics['macro_f1']:.4f} "
                f"DrawP={metrics['draw_precision']:.4f} "
                f"DrawR={metrics['draw_recall']:.4f} "
                f"DrawF1={metrics['draw_f1']:.4f} "
                f"LogLoss={metrics['log_loss']:.4f}"
            )

        for position, (_, row) in enumerate(test_df.iterrows()):
            prediction_rows.append(
                {
                    "window": window_number,
                    "roundNo": row["roundNo"],
                    "homeTeam": row.get("homeTeam", ""),
                    "awayTeam": row.get("awayTeam", ""),
                    "actual": y_test[position],
                    "baseProbA": base_probabilities[position, 0],
                    "baseProbD": base_probabilities[position, 1],
                    "baseProbH": base_probabilities[position, 2],
                    "drawSpecialistProb": draw_probabilities[position],
                }
            )

        print()

        start += args.test_window
        window_number += 1

    actual = all_actual
    base_probabilities = np.vstack(all_base_probabilities)
    draw_probabilities = np.concatenate(all_draw_probabilities)

    alpha_rows = []

    print("=" * 72)
    print("Overall Alpha Comparison")
    print("=" * 72)

    for alpha in alphas:
        adjusted = adjust_draw_probability(
            base_probabilities=base_probabilities,
            specialist_draw_probabilities=draw_probabilities,
            alpha=alpha,
        )

        metrics = calculate_metrics(
            actual=actual,
            probabilities=adjusted,
        )

        alpha_rows.append(
            {
                "alpha": alpha,
                "rows": len(actual),
                "features": len(FEATURE_COLUMNS),
                "rf_weight": rf_weight,
                "lgbm_weight": lgbm_weight,
                **metrics,
            }
        )

    alpha_df = pd.DataFrame(alpha_rows)

    # 総合順位：
    # Accuracyを重視しつつ、Macro F1・Draw F1・LogLossも評価
    alpha_df["selection_score"] = (
        alpha_df["accuracy"] * 0.45
        + alpha_df["macro_f1"] * 0.25
        + alpha_df["draw_f1"] * 0.20
        + (1.0 / (1.0 + alpha_df["log_loss"])) * 0.10
    )

    alpha_df = alpha_df.sort_values(
        [
            "selection_score",
            "accuracy",
            "draw_f1",
            "log_loss",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    print(
        alpha_df[
            [
                "alpha",
                "accuracy",
                "macro_f1",
                "log_loss",
                "draw_precision",
                "draw_recall",
                "draw_f1",
                "predicted_draws",
                "selection_score",
            ]
        ].to_string(index=False)
    )

    best = alpha_df.iloc[0]
    best_alpha = float(best["alpha"])

    print()
    print("=" * 72)
    print("Selected Draw Adjustment")
    print("=" * 72)
    print(f"Best Alpha       : {best_alpha:.2f}")
    print(f"Accuracy         : {best['accuracy']:.4f}")
    print(f"Macro F1         : {best['macro_f1']:.4f}")
    print(f"Log Loss         : {best['log_loss']:.4f}")
    print(f"Draw Precision   : {best['draw_precision']:.4f}")
    print(f"Draw Recall      : {best['draw_recall']:.4f}")
    print(f"Draw F1          : {best['draw_f1']:.4f}")
    print(f"Predicted Draws  : {int(best['predicted_draws'])}")

    # 将来予測用として全データで学習
    x_full = df[FEATURE_COLUMNS].fillna(0)
    y_full = df["result"].astype(str)
    y_full_draw = (y_full == "D").astype(int)

    final_draw_model = create_draw_classifier()
    final_draw_model.fit(x_full, y_full_draw)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "draw_model": final_draw_model,
            "features": FEATURE_COLUMNS,
            "alpha": best_alpha,
            "rf_weight": rf_weight,
            "lgbm_weight": lgbm_weight,
            "metrics": {
                "accuracy": float(best["accuracy"]),
                "macro_f1": float(best["macro_f1"]),
                "log_loss": float(best["log_loss"]),
                "draw_precision": float(best["draw_precision"]),
                "draw_recall": float(best["draw_recall"]),
                "draw_f1": float(best["draw_f1"]),
            },
        },
        MODEL_PATH,
    )

    pd.DataFrame(window_rows).to_csv(
        OUTPUT_WINDOW_RESULTS,
        index=False,
        encoding="utf-8-sig",
    )

    alpha_df.to_csv(
        OUTPUT_ALPHA_RESULTS,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(prediction_rows).to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Saved model : {MODEL_PATH}")
    print(f"Saved       : {OUTPUT_ALPHA_RESULTS}")
    print(f"Saved       : {OUTPUT_WINDOW_RESULTS}")
    print(f"Saved       : {OUTPUT_PREDICTIONS}")


if __name__ == "__main__":
    main()