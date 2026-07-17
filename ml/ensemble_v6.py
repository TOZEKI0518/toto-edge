from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

LOGGER = logging.getLogger("ensemble_v6")

CLASS_ORDER = ["A", "D", "H"]
PROBABILITY_COLUMNS = ["prob_away", "prob_draw", "prob_home"]
KEY_COLUMNS = ["match_card_id"]
METADATA_COLUMNS = [
    "window",
    "match_card_id",
    "match_date",
    "season",
    "competition",
    "home_team",
    "away_team",
    "actual",
]


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """Input and output paths for Ensemble v6."""

    rf_predictions_csv: Path
    lgbm_predictions_csv: Path
    output_dir: Path

    def validate(self) -> None:
        for label, path in (
            ("RandomForest predictions", self.rf_predictions_csv),
            ("LightGBM predictions", self.lgbm_predictions_csv),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
            if not path.is_file():
                raise ValueError(f"{label} is not a file: {path}")


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    """Evaluation metrics for one probability-combination strategy."""

    rows: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    log_loss: float
    away_precision: float
    away_recall: float
    away_f1: float
    draw_precision: float
    draw_recall: float
    draw_f1: float
    home_precision: float
    home_recall: float
    home_f1: float
    predicted_away: int
    predicted_draw: int
    predicted_home: int


@dataclass(frozen=True, slots=True)
class StrategyDefinition:
    """Named probability-combination strategy."""

    name: str
    description: str
    combine: Callable[[np.ndarray, np.ndarray], np.ndarray]


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    """Load and validate one model's rolling-backtest predictions."""
    frame = pd.read_csv(path)
    required = set(METADATA_COLUMNS + PROBABILITY_COLUMNS + ["prediction"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{model_name} prediction CSV missing: {missing}")

    if frame.empty:
        raise ValueError(f"{model_name} prediction CSV is empty: {path}")

    duplicate_ids = int(frame["match_card_id"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(
            f"{model_name} predictions contain {duplicate_ids} duplicate IDs."
        )

    frame = frame.copy()
    frame["actual"] = frame["actual"].astype(str).str.strip().str.upper()
    frame["prediction"] = (
        frame["prediction"].astype(str).str.strip().str.upper()
    )

    invalid_actual = ~frame["actual"].isin(CLASS_ORDER)
    if invalid_actual.any():
        raise ValueError(
            f"{model_name} contains {int(invalid_actual.sum())} invalid actual labels."
        )

    probabilities = frame[PROBABILITY_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if probabilities.isna().any().any():
        raise ValueError(f"{model_name} contains missing probability values.")
    if not np.isfinite(probabilities.to_numpy()).all():
        raise ValueError(f"{model_name} contains non-finite probabilities.")
    if (probabilities.to_numpy() < 0).any():
        raise ValueError(f"{model_name} contains negative probabilities.")

    frame[PROBABILITY_COLUMNS] = normalize_probabilities(
        probabilities.to_numpy(dtype=float)
    )

    LOGGER.info(
        "Loaded %s predictions: rows=%d path=%s",
        model_name,
        len(frame),
        path,
    )
    return frame


def align_models(
    rf: pd.DataFrame,
    lgbm: pd.DataFrame,
) -> pd.DataFrame:
    """Align RF and LightGBM rows and verify identical targets/metadata."""
    rf_columns = METADATA_COLUMNS + PROBABILITY_COLUMNS
    lgbm_columns = ["match_card_id"] + PROBABILITY_COLUMNS

    merged = rf[rf_columns].merge(
        lgbm[lgbm_columns],
        on="match_card_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_rf", "_lgbm"),
        sort=False,
    )

    if len(merged) != len(rf) or len(merged) != len(lgbm):
        rf_only = set(rf["match_card_id"]) - set(lgbm["match_card_id"])
        lgbm_only = set(lgbm["match_card_id"]) - set(rf["match_card_id"])
        raise ValueError(
            "Prediction row mismatch. "
            f"RF rows={len(rf)}, LGBM rows={len(lgbm)}, merged={len(merged)}, "
            f"RF-only examples={list(rf_only)[:10]}, "
            f"LGBM-only examples={list(lgbm_only)[:10]}"
        )

    lgbm_check = lgbm.set_index("match_card_id")
    for column in ["actual", "window", "match_date"]:
        expected = merged["match_card_id"].map(lgbm_check[column])
        actual = merged[column]
        if column == "actual":
            expected = expected.astype(str).str.upper()
            actual = actual.astype(str).str.upper()
        mismatch = actual.astype(str).to_numpy() != expected.astype(str).to_numpy()
        if mismatch.any():
            raise ValueError(
                f"RF and LGBM disagree on {column} for "
                f"{int(mismatch.sum())} rows."
            )

    return merged.sort_values(
        ["window", "match_date", "match_card_id"],
        kind="stable",
    ).reset_index(drop=True)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clip and normalize probability rows."""
    result = np.asarray(probabilities, dtype=float).copy()
    result = np.clip(result, 0.0, None)
    sums = result.sum(axis=1, keepdims=True)
    invalid = sums.squeeze() <= 0
    result[~invalid] = result[~invalid] / sums[~invalid]
    result[invalid] = 1.0 / len(CLASS_ORDER)
    return result


def fixed_blend(rf_weight: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a fixed RF/LGBM weighted-average combiner."""
    if not 0.0 <= rf_weight <= 1.0:
        raise ValueError("rf_weight must be between 0 and 1.")

    def combine(rf: np.ndarray, lgbm: np.ndarray) -> np.ndarray:
        return normalize_probabilities(
            rf_weight * rf + (1.0 - rf_weight) * lgbm
        )

    return combine


def class_specific_blend(
    away_rf_weight: float,
    draw_rf_weight: float,
    home_rf_weight: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a class-specific RF/LGBM probability combiner."""
    weights = np.array(
        [away_rf_weight, draw_rf_weight, home_rf_weight],
        dtype=float,
    )
    if ((weights < 0) | (weights > 1)).any():
        raise ValueError("Class-specific RF weights must be between 0 and 1.")

    def combine(rf: np.ndarray, lgbm: np.ndarray) -> np.ndarray:
        combined = rf * weights + lgbm * (1.0 - weights)
        return normalize_probabilities(combined)

    return combine


def dynamic_blend(
    rf: np.ndarray,
    lgbm: np.ndarray,
) -> np.ndarray:
    """Blend per match using confidence and model-specific class strengths.

    Base class weights favor LightGBM on draws and RandomForest on home wins.
    When the models strongly agree, the blend becomes more balanced. When one
    model is materially more confident, its contribution is increased without
    allowing a single model to dominate completely.
    """
    base_rf_weights = np.array([0.50, 0.30, 0.65], dtype=float)
    combined = rf * base_rf_weights + lgbm * (1.0 - base_rf_weights)

    rf_confidence = rf.max(axis=1)
    lgbm_confidence = lgbm.max(axis=1)
    rf_pick = rf.argmax(axis=1)
    lgbm_pick = lgbm.argmax(axis=1)
    agreement = rf_pick == lgbm_pick

    # Agreement: use the mean, which generally improves calibration.
    combined[agreement] = 0.5 * rf[agreement] + 0.5 * lgbm[agreement]

    # Disagreement: modestly favor the more confident model if the gap is clear.
    disagreement = ~agreement
    confidence_gap = rf_confidence - lgbm_confidence
    favor_rf = disagreement & (confidence_gap >= 0.08)
    favor_lgbm = disagreement & (confidence_gap <= -0.08)

    combined[favor_rf] = 0.65 * rf[favor_rf] + 0.35 * lgbm[favor_rf]
    combined[favor_lgbm] = 0.35 * rf[favor_lgbm] + 0.65 * lgbm[favor_lgbm]

    # Preserve LightGBM's stronger draw behavior when its draw signal is clear.
    draw_signal = (
        (lgbm[:, 1] >= 0.34)
        & (lgbm[:, 1] >= rf[:, 1] + 0.04)
    )
    combined[draw_signal, 1] = (
        0.20 * rf[draw_signal, 1]
        + 0.80 * lgbm[draw_signal, 1]
    )

    # Preserve RF's stronger home signal when it is both high and decisive.
    home_signal = (
        (rf[:, 2] >= 0.55)
        & (rf[:, 2] >= lgbm[:, 2] + 0.05)
    )
    combined[home_signal, 2] = (
        0.80 * rf[home_signal, 2]
        + 0.20 * lgbm[home_signal, 2]
    )

    return normalize_probabilities(combined)


def strategy_definitions() -> list[StrategyDefinition]:
    """Return predefined, reproducible Ensemble v6 candidates."""
    return [
        StrategyDefinition(
            "rf_100",
            "RandomForest v6 only",
            fixed_blend(1.00),
        ),
        StrategyDefinition(
            "lgbm_100",
            "LightGBM v6 only",
            fixed_blend(0.00),
        ),
        StrategyDefinition(
            "fixed_rf60_lgbm40",
            "Fixed blend: RF 60%, LGBM 40%",
            fixed_blend(0.60),
        ),
        StrategyDefinition(
            "fixed_rf55_lgbm45",
            "Fixed blend: RF 55%, LGBM 45%",
            fixed_blend(0.55),
        ),
        StrategyDefinition(
            "fixed_rf50_lgbm50",
            "Fixed blend: RF 50%, LGBM 50%",
            fixed_blend(0.50),
        ),
        StrategyDefinition(
            "fixed_rf45_lgbm55",
            "Fixed blend: RF 45%, LGBM 55%",
            fixed_blend(0.45),
        ),
        StrategyDefinition(
            "fixed_rf40_lgbm60",
            "Fixed blend: RF 40%, LGBM 60%",
            fixed_blend(0.40),
        ),
        StrategyDefinition(
            "class_balanced",
            "Class blend: Away RF50%, Draw RF25%, Home RF65%",
            class_specific_blend(0.50, 0.25, 0.65),
        ),
        StrategyDefinition(
            "class_draw_focus",
            "Class blend: Away RF45%, Draw RF15%, Home RF70%",
            class_specific_blend(0.45, 0.15, 0.70),
        ),
        StrategyDefinition(
            "dynamic_v6",
            "Dynamic confidence blend with draw/home specialization",
            dynamic_blend,
        ),
    ]


def evaluate(
    actual: pd.Series,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, StrategyMetrics]:
    """Evaluate one probability matrix."""
    probabilities = normalize_probabilities(probabilities)
    predicted_indices = probabilities.argmax(axis=1)
    predicted = np.array(CLASS_ORDER, dtype=object)[predicted_indices]

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=CLASS_ORDER,
        zero_division=0,
    )
    counts = pd.Series(predicted).value_counts()

    metrics = StrategyMetrics(
        rows=len(actual),
        accuracy=float(accuracy_score(actual, predicted)),
        macro_f1=float(f1_score(actual, predicted, average="macro")),
        weighted_f1=float(
            f1_score(actual, predicted, average="weighted")
        ),
        log_loss=float(log_loss(actual, probabilities, labels=CLASS_ORDER)),
        away_precision=float(precision[0]),
        away_recall=float(recall[0]),
        away_f1=float(f1[0]),
        draw_precision=float(precision[1]),
        draw_recall=float(recall[1]),
        draw_f1=float(f1[1]),
        home_precision=float(precision[2]),
        home_recall=float(recall[2]),
        home_f1=float(f1[2]),
        predicted_away=int(counts.get("A", 0)),
        predicted_draw=int(counts.get("D", 0)),
        predicted_home=int(counts.get("H", 0)),
    )
    return predicted, metrics


def run_strategies(
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Evaluate every strategy and return comparison/predictions/window metrics."""
    rf = aligned[
        [f"{column}_rf" for column in PROBABILITY_COLUMNS]
    ].to_numpy(dtype=float)
    lgbm = aligned[
        [f"{column}_lgbm" for column in PROBABILITY_COLUMNS]
    ].to_numpy(dtype=float)
    actual = aligned["actual"].astype(str).str.upper()

    comparison_rows: list[dict[str, object]] = []
    prediction_outputs: dict[str, pd.DataFrame] = {}
    window_rows: list[dict[str, object]] = []

    for strategy in strategy_definitions():
        probabilities = strategy.combine(rf, lgbm)
        predicted, metrics = evaluate(actual, probabilities)

        comparison_rows.append(
            {
                "strategy": strategy.name,
                "description": strategy.description,
                **asdict(metrics),
            }
        )

        output = aligned[METADATA_COLUMNS].copy()
        output["prediction"] = predicted
        output["prob_away"] = probabilities[:, 0]
        output["prob_draw"] = probabilities[:, 1]
        output["prob_home"] = probabilities[:, 2]
        output["rf_prob_away"] = rf[:, 0]
        output["rf_prob_draw"] = rf[:, 1]
        output["rf_prob_home"] = rf[:, 2]
        output["lgbm_prob_away"] = lgbm[:, 0]
        output["lgbm_prob_draw"] = lgbm[:, 1]
        output["lgbm_prob_home"] = lgbm[:, 2]
        output["strategy"] = strategy.name
        prediction_outputs[strategy.name] = output

        for window, window_frame in output.groupby("window", sort=True):
            window_probabilities = window_frame[PROBABILITY_COLUMNS].to_numpy()
            _, window_metrics = evaluate(
                window_frame["actual"],
                window_probabilities,
            )
            window_rows.append(
                {
                    "strategy": strategy.name,
                    "window": int(window),
                    "test_from": str(window_frame["match_date"].min()),
                    "test_to": str(window_frame["match_date"].max()),
                    **asdict(window_metrics),
                }
            )

        LOGGER.info(
            "%s | Accuracy=%.6f MacroF1=%.6f LogLoss=%.6f DrawF1=%.6f",
            strategy.name,
            metrics.accuracy,
            metrics.macro_f1,
            metrics.log_loss,
            metrics.draw_f1,
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison["accuracy_rank"] = comparison["accuracy"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison["macro_f1_rank"] = comparison["macro_f1"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison["log_loss_rank"] = comparison["log_loss"].rank(
        method="min", ascending=True
    ).astype(int)
    comparison["composite_rank_score"] = (
        comparison["accuracy_rank"]
        + comparison["macro_f1_rank"]
        + comparison["log_loss_rank"]
    )
    comparison = comparison.sort_values(
        ["composite_rank_score", "accuracy", "macro_f1"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    windows = pd.DataFrame(window_rows)
    return comparison, prediction_outputs, windows


def choose_best_strategy(comparison: pd.DataFrame) -> str:
    """Choose the best balanced strategy by three core metric ranks."""
    return str(comparison.iloc[0]["strategy"])


def save_outputs(
    config: EnsembleConfig,
    comparison: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    windows: pd.DataFrame,
) -> str:
    """Save comparison, all-strategy results and best-strategy artifacts."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = config.output_dir / "ensemble_v6_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    windows.to_csv(
        config.output_dir / "ensemble_v6_by_window.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_strategy = choose_best_strategy(comparison)
    best_predictions = predictions[best_strategy]
    best_predictions.to_csv(
        config.output_dir / "ensemble_v6_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_predictions = pd.concat(
        predictions.values(),
        ignore_index=True,
    )
    all_predictions.to_csv(
        config.output_dir / "ensemble_v6_all_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_row = comparison.loc[
        comparison["strategy"] == best_strategy
    ].iloc[0]
    summary = {
        "best_strategy": best_strategy,
        "selection_method": (
            "minimum sum of Accuracy rank, Macro F1 rank, and Log Loss rank"
        ),
        "rf_predictions_csv": str(config.rf_predictions_csv),
        "lgbm_predictions_csv": str(config.lgbm_predictions_csv),
        "metrics": {
            key: value.item() if hasattr(value, "item") else value
            for key, value in best_row.to_dict().items()
        },
    }
    (config.output_dir / "ensemble_v6_best_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return best_strategy


def print_summary(
    comparison: pd.DataFrame,
    best_strategy: str,
    output_dir: Path,
) -> None:
    columns = [
        "strategy",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "away_f1",
        "draw_f1",
        "home_f1",
        "predicted_away",
        "predicted_draw",
        "predicted_home",
        "composite_rank_score",
    ]

    print("=" * 132)
    print("Project Alpha Ensemble v6")
    print("=" * 132)
    print(
        comparison[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"Best balanced strategy : {best_strategy}")
    print(f"Outputs                : {output_dir}")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Compare RF v6 and LightGBM v6 fixed, class-specific, and dynamic "
            "probability ensembles."
        )
    )
    parser.add_argument(
        "--rf-predictions",
        type=Path,
        default=root / "ml" / "rf_v6" / "randomforest_v6_predictions.csv",
    )
    parser.add_argument(
        "--lgbm-predictions",
        type=Path,
        default=root / "ml" / "lgbm_v6" / "lgbm_v6_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "ml" / "ensemble_v6",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    config = EnsembleConfig(
        rf_predictions_csv=args.rf_predictions,
        lgbm_predictions_csv=args.lgbm_predictions,
        output_dir=args.output_dir,
    )
    config.validate()

    rf = load_predictions(config.rf_predictions_csv, "RandomForest v6")
    lgbm = load_predictions(config.lgbm_predictions_csv, "LightGBM v6")
    aligned = align_models(rf, lgbm)

    comparison, predictions, windows = run_strategies(aligned)
    best_strategy = save_outputs(
        config=config,
        comparison=comparison,
        predictions=predictions,
        windows=windows,
    )
    print_summary(comparison, best_strategy, config.output_dir)


if __name__ == "__main__":
    main()
