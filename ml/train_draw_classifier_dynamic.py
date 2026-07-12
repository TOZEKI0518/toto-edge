from __future__ import annotations

from itertools import product
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

OUTPUT_CONFIG_RESULTS = Path("ml/draw_dynamic_config_results.csv")
OUTPUT_WINDOW_RESULTS = Path("ml/draw_dynamic_window_results.csv")
OUTPUT_PREDICTIONS = Path("ml/draw_dynamic_predictions.csv")
MODEL_PATH = Path("ml/toto_draw_classifier_dynamic.joblib")


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

    model_class_list = list(model_classes)

    for target_index, target_class in enumerate(TARGET_CLASSES):
        if target_class in model_class_list:
            source_index = model_class_list.index(target_class)
            aligned[:, target_index] = probabilities[:, source_index]

    row_sums = aligned.sum(axis=1, keepdims=True)

    zero_rows = row_sums[:, 0] <= 0
    if zero_rows.any():
        aligned[zero_rows] = 1.0 / len(TARGET_CLASSES)
        row_sums = aligned.sum(axis=1, keepdims=True)

    return aligned / row_sums


def labels_from_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    indexes = probabilities.argmax(axis=1)

    return np.array(
        [TARGET_CLASSES[index] for index in indexes],
        dtype=object,
    )


def calculate_dynamic_alpha(
    draw_probabilities: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    low_alpha: float,
    medium_alpha: float,
    high_alpha: float,
) -> np.ndarray:
    """
    Draw専門モデルの確率に応じて補正強度を変更する。

    draw_prob < low_threshold
        -> low_alpha

    low_threshold <= draw_prob < high_threshold
        -> medium_alpha

    draw_prob >= high_threshold
        -> high_alpha
    """
    alpha = np.full(
        shape=len(draw_probabilities),
        fill_value=low_alpha,
        dtype=float,
    )

    medium_mask = (
        (draw_probabilities >= low_threshold)
        & (draw_probabilities < high_threshold)
    )

    high_mask = draw_probabilities >= high_threshold

    alpha[medium_mask] = medium_alpha
    alpha[high_mask] = high_alpha

    return alpha


def adjust_draw_probability_dynamic(
    base_probabilities: np.ndarray,
    specialist_draw_probabilities: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    low_alpha: float,
    medium_alpha: float,
    high_alpha: float,
    minimum_draw_probability: float = 0.02,
    maximum_draw_probability: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = calculate_dynamic_alpha(
        draw_probabilities=specialist_draw_probabilities,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        low_alpha=low_alpha,
        medium_alpha=medium_alpha,
        high_alpha=high_alpha,
    )

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

    safe_non_draw_total = np.where(
        non_draw_total > 0,
        non_draw_total,
        1.0,
    )

    adjusted = np.zeros_like(base_probabilities)

    adjusted[:, 0] = remaining * (
        base_a / safe_non_draw_total
    )
    adjusted[:, 1] = new_d
    adjusted[:, 2] = remaining * (
        base_h / safe_non_draw_total
    )

    zero_non_draw = non_draw_total <= 0

    if zero_non_draw.any():
        adjusted[zero_non_draw, 0] = (
            remaining[zero_non_draw] / 2
        )
        adjusted[zero_non_draw, 2] = (
            remaining[zero_non_draw] / 2
        )

    row_sums = adjusted.sum(axis=1, keepdims=True)

    return adjusted / row_sums, alpha


def calculate_metrics(
    actual: list[str],
    probabilities: np.ndarray,
) -> dict:
    predicted = labels_from_probabilities(probabilities)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=TARGET_CLASSES,
            zero_division=0,
        )
    )

    draw_index = TARGET_CLASSES.index("D")

    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=TARGET_CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual,
                predicted,
                labels=TARGET_CLASSES,
                average="weighted",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                actual,
                probabilities,
                labels=TARGET_CLASSES,
            )
        ),
        "draw_precision": float(
            precision[draw_index]
        ),
        "draw_recall": float(
            recall[draw_index]
        ),
        "draw_f1": float(
            f1[draw_index]
        ),
        "draw_support": int(
            support[draw_index]
        ),
        "predicted_draws": int(
            (predicted == "D").sum()
        ),
    }


def build_configurations():
    low_thresholds = [0.30, 0.35]
    high_thresholds = [0.40, 0.45, 0.50]

    low_alphas = [0.00, 0.05]
    medium_alphas = [0.10, 0.15, 0.20]
    high_alphas = [0.25, 0.30, 0.35, 0.40]

    configs = []

    for (
        low_threshold,
        high_threshold,
        low_alpha,
        medium_alpha,
        high_alpha,
    ) in product(
        low_thresholds,
        high_thresholds,
        low_alphas,
        medium_alphas,
        high_alphas,
    ):
        if low_threshold >= high_threshold:
            continue

        if not (
            low_alpha
            <= medium_alpha
            <= high_alpha
        ):
            continue

        configs.append(
            {
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "low_alpha": low_alpha,
                "medium_alpha": medium_alpha,
                "high_alpha": high_alpha,
            }
        )

    return configs


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
        "--min-accuracy",
        type=float,
        default=0.44,
        help=(
            "正式候補に残す最低Accuracy。"
            "該当候補がなければ総合スコア最大を選ぶ。"
        ),
    )

    args = parser.parse_args()

    weight_total = (
        args.rf_weight + args.lgbm_weight
    )

    if weight_total <= 0:
        raise ValueError(
            "Model weights must total more than zero."
        )

    rf_weight = args.rf_weight / weight_total
    lgbm_weight = args.lgbm_weight / weight_total

    configurations = build_configurations()

    df = load_dataset()
    df = df[df["result"].isin(TARGET_CLASSES)].copy()
    df = df.sort_values("roundNo").reset_index(drop=True)

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    rounds = sorted(df["roundNo"].unique())

    if len(rounds) <= args.min_train_rounds:
        raise ValueError(
            "Not enough rounds for rolling evaluation."
        )

    all_actual = []
    all_base_probabilities = []
    all_draw_probabilities = []

    window_rows = []
    prediction_rows = []

    start = args.min_train_rounds
    window_number = 1

    print("=" * 72)
    print("Dynamic Draw Classifier Rolling Search")
    print("=" * 72)
    print(f"Rows              : {len(df)}")
    print(
        f"Rounds            : "
        f"{rounds[0]} - {rounds[-1]}"
    )
    print(
        f"Features          : "
        f"{len(FEATURE_COLUMNS)}"
    )
    print(
        f"Base Ensemble     : "
        f"RF {rf_weight:.2f} / "
        f"LightGBM {lgbm_weight:.2f}"
    )
    print(
        f"Configurations    : "
        f"{len(configurations)}"
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

        x_train = train_df[
            FEATURE_COLUMNS
        ].fillna(0)

        x_test = test_df[
            FEATURE_COLUMNS
        ].fillna(0)

        y_train = train_df["result"].astype(str)
        y_test = test_df["result"].astype(str).tolist()

        y_train_draw = (
            y_train == "D"
        ).astype(int)

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

        draw_probabilities = (
            draw_model.predict_proba(x_test)[:, 1]
        )

        all_actual.extend(y_test)
        all_base_probabilities.append(
            base_probabilities
        )
        all_draw_probabilities.append(
            draw_probabilities
        )

        base_metrics = calculate_metrics(
            actual=y_test,
            probabilities=base_probabilities,
        )

        print(
            f"Window {window_number}: "
            f"Train {train_rounds[0]}-"
            f"{train_rounds[-1]} "
            f"({len(train_df)} rows) -> "
            f"Test {test_rounds[0]}-"
            f"{test_rounds[-1]} "
            f"({len(test_df)} rows) | "
            f"Base Accuracy="
            f"{base_metrics['accuracy']:.4f} "
            f"DrawRecall="
            f"{base_metrics['draw_recall']:.4f}"
        )

        for position, (_, row) in enumerate(
            test_df.iterrows()
        ):
            prediction_rows.append(
                {
                    "window": window_number,
                    "roundNo": row["roundNo"],
                    "homeTeam": row.get(
                        "homeTeam",
                        "",
                    ),
                    "awayTeam": row.get(
                        "awayTeam",
                        "",
                    ),
                    "actual": y_test[position],
                    "baseProbA": (
                        base_probabilities[position, 0]
                    ),
                    "baseProbD": (
                        base_probabilities[position, 1]
                    ),
                    "baseProbH": (
                        base_probabilities[position, 2]
                    ),
                    "drawSpecialistProb": (
                        draw_probabilities[position]
                    ),
                }
            )

        start += args.test_window
        window_number += 1

    actual = all_actual

    base_probabilities = np.vstack(
        all_base_probabilities
    )

    draw_probabilities = np.concatenate(
        all_draw_probabilities
    )

    config_rows = []

    print()
    print("=" * 72)
    print("Dynamic Configuration Evaluation")
    print("=" * 72)

    for config_number, config in enumerate(
        configurations,
        start=1,
    ):
        adjusted, alpha_values = (
            adjust_draw_probability_dynamic(
                base_probabilities=base_probabilities,
                specialist_draw_probabilities=draw_probabilities,
                **config,
            )
        )

        metrics = calculate_metrics(
            actual=actual,
            probabilities=adjusted,
        )

        low_count = int(
            (
                draw_probabilities
                < config["low_threshold"]
            ).sum()
        )

        medium_count = int(
            (
                (
                    draw_probabilities
                    >= config["low_threshold"]
                )
                & (
                    draw_probabilities
                    < config["high_threshold"]
                )
            ).sum()
        )

        high_count = int(
            (
                draw_probabilities
                >= config["high_threshold"]
            ).sum()
        )

        config_rows.append(
            {
                "config_id": config_number,
                **config,
                "rows": len(actual),
                "low_count": low_count,
                "medium_count": medium_count,
                "high_count": high_count,
                "average_alpha": float(
                    alpha_values.mean()
                ),
                **metrics,
            }
        )

    results_df = pd.DataFrame(config_rows)

    base_metrics = calculate_metrics(
        actual=actual,
        probabilities=base_probabilities,
    )

    results_df["accuracy_change"] = (
        results_df["accuracy"]
        - base_metrics["accuracy"]
    )

    results_df["draw_recall_change"] = (
        results_df["draw_recall"]
        - base_metrics["draw_recall"]
    )

    results_df["draw_f1_change"] = (
        results_df["draw_f1"]
        - base_metrics["draw_f1"]
    )

    results_df["selection_score"] = (
        results_df["accuracy"] * 0.55
        + results_df["macro_f1"] * 0.20
        + results_df["draw_f1"] * 0.15
        + (
            1.0
            / (1.0 + results_df["log_loss"])
        ) * 0.10
    )

    eligible = results_df[
        results_df["accuracy"]
        >= args.min_accuracy
    ].copy()

    if eligible.empty:
        print(
            "No configuration met the minimum "
            "accuracy requirement. "
            "Using highest overall score."
        )
        ranked = results_df.copy()
    else:
        ranked = eligible

    ranked = ranked.sort_values(
        [
            "selection_score",
            "accuracy",
            "draw_f1",
            "draw_recall",
            "log_loss",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    best = ranked.iloc[0]

    display_columns = [
        "config_id",
        "low_threshold",
        "high_threshold",
        "low_alpha",
        "medium_alpha",
        "high_alpha",
        "accuracy",
        "accuracy_change",
        "macro_f1",
        "log_loss",
        "draw_precision",
        "draw_recall",
        "draw_recall_change",
        "draw_f1",
        "draw_f1_change",
        "predicted_draws",
        "selection_score",
    ]

    print()
    print("Top 15 Configurations")
    print(
        ranked[
            display_columns
        ].head(15).to_string(index=False)
    )

    print()
    print("=" * 72)
    print("Selected Dynamic Draw Configuration")
    print("=" * 72)
    print(
        f"Config ID          : "
        f"{int(best['config_id'])}"
    )
    print(
        f"Low threshold      : "
        f"{best['low_threshold']:.2f}"
    )
    print(
        f"High threshold     : "
        f"{best['high_threshold']:.2f}"
    )
    print(
        f"Low alpha          : "
        f"{best['low_alpha']:.2f}"
    )
    print(
        f"Medium alpha       : "
        f"{best['medium_alpha']:.2f}"
    )
    print(
        f"High alpha         : "
        f"{best['high_alpha']:.2f}"
    )
    print()
    print(
        f"Accuracy           : "
        f"{best['accuracy']:.4f} "
        f"({best['accuracy_change']:+.4f})"
    )
    print(
        f"Macro F1           : "
        f"{best['macro_f1']:.4f}"
    )
    print(
        f"Log Loss           : "
        f"{best['log_loss']:.4f}"
    )
    print(
        f"Draw Precision     : "
        f"{best['draw_precision']:.4f}"
    )
    print(
        f"Draw Recall        : "
        f"{best['draw_recall']:.4f} "
        f"({best['draw_recall_change']:+.4f})"
    )
    print(
        f"Draw F1            : "
        f"{best['draw_f1']:.4f} "
        f"({best['draw_f1_change']:+.4f})"
    )
    print(
        f"Predicted Draws    : "
        f"{int(best['predicted_draws'])}"
    )

    x_full = df[
        FEATURE_COLUMNS
    ].fillna(0)

    y_full = df["result"].astype(str)
    y_full_draw = (
        y_full == "D"
    ).astype(int)

    final_draw_model = create_draw_classifier()
    final_draw_model.fit(
        x_full,
        y_full_draw,
    )

    model_payload = {
        "draw_model": final_draw_model,
        "features": FEATURE_COLUMNS,
        "rf_weight": rf_weight,
        "lgbm_weight": lgbm_weight,
        "low_threshold": float(
            best["low_threshold"]
        ),
        "high_threshold": float(
            best["high_threshold"]
        ),
        "low_alpha": float(
            best["low_alpha"]
        ),
        "medium_alpha": float(
            best["medium_alpha"]
        ),
        "high_alpha": float(
            best["high_alpha"]
        ),
        "metrics": {
            "accuracy": float(
                best["accuracy"]
            ),
            "macro_f1": float(
                best["macro_f1"]
            ),
            "log_loss": float(
                best["log_loss"]
            ),
            "draw_precision": float(
                best["draw_precision"]
            ),
            "draw_recall": float(
                best["draw_recall"]
            ),
            "draw_f1": float(
                best["draw_f1"]
            ),
        },
        "base_metrics": base_metrics,
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_payload,
        MODEL_PATH,
    )

    results_df.sort_values(
        [
            "selection_score",
            "accuracy",
        ],
        ascending=[
            False,
            False,
        ],
    ).to_csv(
        OUTPUT_CONFIG_RESULTS,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        prediction_rows
    ).to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
        encoding="utf-8-sig",
    )

    # 各Windowで選択設定を評価
    start_index = 0

    for window_number, probabilities in enumerate(
        all_base_probabilities,
        start=1,
    ):
        row_count = len(probabilities)

        window_actual = actual[
            start_index : start_index + row_count
        ]

        window_draw_probabilities = (
            all_draw_probabilities[
                window_number - 1
            ]
        )

        adjusted, alpha_values = (
            adjust_draw_probability_dynamic(
                base_probabilities=probabilities,
                specialist_draw_probabilities=window_draw_probabilities,
                low_threshold=float(
                    best["low_threshold"]
                ),
                high_threshold=float(
                    best["high_threshold"]
                ),
                low_alpha=float(
                    best["low_alpha"]
                ),
                medium_alpha=float(
                    best["medium_alpha"]
                ),
                high_alpha=float(
                    best["high_alpha"]
                ),
            )
        )

        metrics = calculate_metrics(
            actual=window_actual,
            probabilities=adjusted,
        )

        window_rows.append(
            {
                "window": window_number,
                "rows": row_count,
                "average_alpha": float(
                    alpha_values.mean()
                ),
                **metrics,
            }
        )

        start_index += row_count

    pd.DataFrame(
        window_rows
    ).to_csv(
        OUTPUT_WINDOW_RESULTS,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Saved model : {MODEL_PATH}")
    print(f"Saved       : {OUTPUT_CONFIG_RESULTS}")
    print(f"Saved       : {OUTPUT_WINDOW_RESULTS}")
    print(f"Saved       : {OUTPUT_PREDICTIONS}")


if __name__ == "__main__":
    main()