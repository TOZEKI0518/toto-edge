from __future__ import annotations

from pathlib import Path
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
)

from common.dataset import load_dataset
from common.features import FEATURE_COLUMNS
from common.models import create_random_forest


TARGET_CLASSES = ["A", "D", "H"]

OUTPUT_SUMMARY = Path("ml/calibration_summary.csv")
OUTPUT_WINDOWS = Path("ml/calibration_windows.csv")
OUTPUT_PREDICTIONS = Path("ml/calibration_predictions.csv")
OUTPUT_CONFIG = Path("ml/calibration_config.json")


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
        (len(probabilities), len(TARGET_CLASSES)),
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


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Temperature > 1:
        確率を平たくし、過信を弱める。

    Temperature < 1:
        確率差を強める。
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    clipped = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    logits = np.log(clipped)
    scaled_logits = logits / temperature

    scaled_logits -= scaled_logits.max(
        axis=1,
        keepdims=True,
    )

    exp_logits = np.exp(scaled_logits)

    return exp_logits / exp_logits.sum(
        axis=1,
        keepdims=True,
    )


def find_best_temperature(
    probabilities: np.ndarray,
    actual: list[str],
    temperatures: np.ndarray,
) -> tuple[float, float]:
    best_temperature = 1.0
    best_loss = float("inf")

    for temperature in temperatures:
        calibrated = apply_temperature(
            probabilities,
            float(temperature),
        )

        loss = log_loss(
            actual,
            calibrated,
            labels=TARGET_CLASSES,
        )

        if loss < best_loss:
            best_loss = float(loss)
            best_temperature = float(temperature)

    return best_temperature, best_loss


def labels_from_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    indexes = probabilities.argmax(axis=1)

    return np.array(
        [TARGET_CLASSES[index] for index in indexes],
        dtype=object,
    )


def multiclass_brier_score(
    actual: list[str],
    probabilities: np.ndarray,
) -> float:
    actual_array = np.array(actual)

    scores = []

    for class_index, class_name in enumerate(TARGET_CLASSES):
        binary_actual = (
            actual_array == class_name
        ).astype(int)

        scores.append(
            brier_score_loss(
                binary_actual,
                probabilities[:, class_index],
            )
        )

    return float(np.mean(scores))


def calculate_metrics(
    actual: list[str],
    probabilities: np.ndarray,
) -> dict:
    predictions = labels_from_probabilities(
        probabilities
    )

    actual_array = np.array(actual)

    draw_mask = actual_array == "D"
    predicted_draw_mask = predictions == "D"

    true_draws = int(
        (draw_mask & predicted_draw_mask).sum()
    )

    actual_draws = int(draw_mask.sum())
    predicted_draws = int(
        predicted_draw_mask.sum()
    )

    draw_recall = (
        true_draws / actual_draws
        if actual_draws
        else 0.0
    )

    draw_precision = (
        true_draws / predicted_draws
        if predicted_draws
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy_score(
                actual,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predictions,
                labels=TARGET_CLASSES,
                average="macro",
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
        "brier_score": multiclass_brier_score(
            actual,
            probabilities,
        ),
        "draw_precision": float(
            draw_precision
        ),
        "draw_recall": float(
            draw_recall
        ),
        "predicted_draws": predicted_draws,
        "average_confidence": float(
            probabilities.max(axis=1).mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-train-rounds",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--calibration-rounds",
        type=int,
        default=10,
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
        "--temperature-min",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--temperature-max",
        type=float,
        default=2.50,
    )

    parser.add_argument(
        "--temperature-step",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    if args.calibration_rounds < 1:
        raise ValueError(
            "--calibration-rounds must be at least 1."
        )

    weight_total = (
        args.rf_weight
        + args.lgbm_weight
    )

    if weight_total <= 0:
        raise ValueError(
            "Model weights must total more than zero."
        )

    rf_weight = (
        args.rf_weight / weight_total
    )

    lgbm_weight = (
        args.lgbm_weight / weight_total
    )

    temperatures = np.arange(
        args.temperature_min,
        args.temperature_max
        + args.temperature_step / 2,
        args.temperature_step,
    )

    df = load_dataset()

    df = df[
        df["result"].isin(TARGET_CLASSES)
    ].copy()

    df = df.sort_values(
        "roundNo"
    ).reset_index(drop=True)

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    rounds = sorted(
        df["roundNo"].unique()
    )

    minimum_required = (
        args.min_train_rounds
        + args.calibration_rounds
    )

    if len(rounds) <= minimum_required:
        raise ValueError(
            "Not enough rounds for training, "
            "calibration and testing."
        )

    all_actual: list[str] = []
    all_raw_probabilities = []
    all_calibrated_probabilities = []

    window_rows = []
    prediction_rows = []
    selected_temperatures = []

    test_start_index = minimum_required
    window_number = 1

    print("=" * 72)
    print("Probability Calibration Rolling Test")
    print("=" * 72)
    print(f"Rows                : {len(df)}")
    print(
        f"Rounds              : "
        f"{rounds[0]} - {rounds[-1]}"
    )
    print(
        f"Features            : "
        f"{len(FEATURE_COLUMNS)}"
    )
    print(
        f"RF / LightGBM       : "
        f"{rf_weight:.2f} / "
        f"{lgbm_weight:.2f}"
    )
    print(
        f"Calibration rounds  : "
        f"{args.calibration_rounds}"
    )
    print(
        f"Temperature grid    : "
        f"{temperatures[0]:.2f} - "
        f"{temperatures[-1]:.2f}"
    )
    print()

    while test_start_index < len(rounds):
        test_rounds = rounds[
            test_start_index:
            test_start_index + args.test_window
        ]

        if not test_rounds:
            break

        calibration_start = (
            test_start_index
            - args.calibration_rounds
        )

        fit_rounds = rounds[
            :calibration_start
        ]

        calibration_rounds = rounds[
            calibration_start:
            test_start_index
        ]

        fit_df = df[
            df["roundNo"].isin(fit_rounds)
        ].copy()

        calibration_df = df[
            df["roundNo"].isin(
                calibration_rounds
            )
        ].copy()

        test_df = df[
            df["roundNo"].isin(test_rounds)
        ].copy()

        x_fit = fit_df[
            FEATURE_COLUMNS
        ].fillna(0)

        x_calibration = calibration_df[
            FEATURE_COLUMNS
        ].fillna(0)

        x_test = test_df[
            FEATURE_COLUMNS
        ].fillna(0)

        y_fit = fit_df[
            "result"
        ].astype(str)

        y_calibration = calibration_df[
            "result"
        ].astype(str).tolist()

        y_test = test_df[
            "result"
        ].astype(str).tolist()

        rf = create_random_forest()
        lgbm_model = create_lightgbm()

        rf.fit(
            x_fit,
            y_fit,
        )

        lgbm_model.fit(
            x_fit,
            y_fit,
        )

        rf_calibration = align_probabilities(
            rf.classes_,
            rf.predict_proba(
                x_calibration
            ),
        )

        lgbm_calibration = align_probabilities(
            lgbm_model.classes_,
            lgbm_model.predict_proba(
                x_calibration
            ),
        )

        calibration_probabilities = (
            rf_calibration * rf_weight
            + lgbm_calibration * lgbm_weight
        )

        best_temperature, calibration_loss = (
            find_best_temperature(
                probabilities=(
                    calibration_probabilities
                ),
                actual=y_calibration,
                temperatures=temperatures,
            )
        )

        selected_temperatures.append(
            best_temperature
        )

        rf_test = align_probabilities(
            rf.classes_,
            rf.predict_proba(x_test),
        )

        lgbm_test = align_probabilities(
            lgbm_model.classes_,
            lgbm_model.predict_proba(
                x_test
            ),
        )

        raw_probabilities = (
            rf_test * rf_weight
            + lgbm_test * lgbm_weight
        )

        calibrated_probabilities = (
            apply_temperature(
                raw_probabilities,
                best_temperature,
            )
        )

        raw_metrics = calculate_metrics(
            actual=y_test,
            probabilities=raw_probabilities,
        )

        calibrated_metrics = calculate_metrics(
            actual=y_test,
            probabilities=(
                calibrated_probabilities
            ),
        )

        window_rows.append(
            {
                "window": window_number,
                "fit_round_start": (
                    fit_rounds[0]
                ),
                "fit_round_end": (
                    fit_rounds[-1]
                ),
                "calibration_round_start": (
                    calibration_rounds[0]
                ),
                "calibration_round_end": (
                    calibration_rounds[-1]
                ),
                "test_round_start": (
                    test_rounds[0]
                ),
                "test_round_end": (
                    test_rounds[-1]
                ),
                "fit_rows": len(fit_df),
                "calibration_rows": len(
                    calibration_df
                ),
                "test_rows": len(test_df),
                "temperature": (
                    best_temperature
                ),
                "calibration_log_loss": (
                    calibration_loss
                ),
                "raw_accuracy": (
                    raw_metrics["accuracy"]
                ),
                "calibrated_accuracy": (
                    calibrated_metrics[
                        "accuracy"
                    ]
                ),
                "raw_log_loss": (
                    raw_metrics["log_loss"]
                ),
                "calibrated_log_loss": (
                    calibrated_metrics[
                        "log_loss"
                    ]
                ),
                "raw_brier": (
                    raw_metrics[
                        "brier_score"
                    ]
                ),
                "calibrated_brier": (
                    calibrated_metrics[
                        "brier_score"
                    ]
                ),
            }
        )

        print(
            f"Window {window_number}: "
            f"Fit {fit_rounds[0]}-"
            f"{fit_rounds[-1]} | "
            f"Cal {calibration_rounds[0]}-"
            f"{calibration_rounds[-1]} | "
            f"Test {test_rounds[0]}-"
            f"{test_rounds[-1]} | "
            f"T={best_temperature:.2f} | "
            f"Raw LL="
            f"{raw_metrics['log_loss']:.4f} | "
            f"Cal LL="
            f"{calibrated_metrics['log_loss']:.4f}"
        )

        for position, (_, row) in enumerate(
            test_df.iterrows()
        ):
            raw_prediction = (
                labels_from_probabilities(
                    raw_probabilities[
                        position:
                        position + 1
                    ]
                )[0]
            )

            calibrated_prediction = (
                labels_from_probabilities(
                    calibrated_probabilities[
                        position:
                        position + 1
                    ]
                )[0]
            )

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
                    "temperature": (
                        best_temperature
                    ),
                    "rawPrediction": (
                        raw_prediction
                    ),
                    "calibratedPrediction": (
                        calibrated_prediction
                    ),
                    "rawProbA": (
                        raw_probabilities[
                            position,
                            0,
                        ]
                    ),
                    "rawProbD": (
                        raw_probabilities[
                            position,
                            1,
                        ]
                    ),
                    "rawProbH": (
                        raw_probabilities[
                            position,
                            2,
                        ]
                    ),
                    "calibratedProbA": (
                        calibrated_probabilities[
                            position,
                            0,
                        ]
                    ),
                    "calibratedProbD": (
                        calibrated_probabilities[
                            position,
                            1,
                        ]
                    ),
                    "calibratedProbH": (
                        calibrated_probabilities[
                            position,
                            2,
                        ]
                    ),
                }
            )

        all_actual.extend(y_test)

        all_raw_probabilities.append(
            raw_probabilities
        )

        all_calibrated_probabilities.append(
            calibrated_probabilities
        )

        test_start_index += (
            args.test_window
        )

        window_number += 1

    raw_all = np.vstack(
        all_raw_probabilities
    )

    calibrated_all = np.vstack(
        all_calibrated_probabilities
    )

    raw_metrics = calculate_metrics(
        actual=all_actual,
        probabilities=raw_all,
    )

    calibrated_metrics = calculate_metrics(
        actual=all_actual,
        probabilities=calibrated_all,
    )

    summary = pd.DataFrame(
        [
            {
                "model": "Ensemble Raw",
                **raw_metrics,
            },
            {
                "model": (
                    "Ensemble Temperature "
                    "Calibrated"
                ),
                **calibrated_metrics,
            },
        ]
    )

    print()
    print("=" * 72)
    print("Calibration Summary")
    print("=" * 72)

    print(
        summary[
            [
                "model",
                "accuracy",
                "macro_f1",
                "log_loss",
                "brier_score",
                "draw_precision",
                "draw_recall",
                "average_confidence",
            ]
        ].to_string(index=False)
    )

    median_temperature = float(
        np.median(
            selected_temperatures
        )
    )

    mean_temperature = float(
        np.mean(
            selected_temperatures
        )
    )

    config = {
        "method": "temperature_scaling",
        "medianTemperature": (
            median_temperature
        ),
        "meanTemperature": (
            mean_temperature
        ),
        "windowTemperatures": (
            selected_temperatures
        ),
        "rfWeight": rf_weight,
        "lgbmWeight": lgbm_weight,
        "rawMetrics": raw_metrics,
        "calibratedMetrics": (
            calibrated_metrics
        ),
    }

    import json

    OUTPUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        window_rows
    ).to_csv(
        OUTPUT_WINDOWS,
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

    OUTPUT_CONFIG.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Median Temperature : "
        f"{median_temperature:.4f}"
    )

    print(
        f"Mean Temperature   : "
        f"{mean_temperature:.4f}"
    )

    print()
    print(f"Saved: {OUTPUT_SUMMARY}")
    print(f"Saved: {OUTPUT_WINDOWS}")
    print(f"Saved: {OUTPUT_PREDICTIONS}")
    print(f"Saved: {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()