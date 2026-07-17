from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger("random_forest_v6")

CLASS_ORDER = ["A", "D", "H"]
TARGET_COLUMN = "result"
EXCLUDED_COLUMNS = {
    "match_card_id",
    "match_date",
    "home_team",
    "away_team",
    "season",
    "competition",
    "round",
    "home_score",
    "away_score",
    TARGET_COLUMN,
}


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """RandomForest v6 rolling-backtest configuration."""

    input_csv: Path
    output_dir: Path
    min_train_days: int = 365
    test_window_days: int = 90
    random_state: int = 42

    def validate(self) -> None:
        if not self.input_csv.exists():
            raise FileNotFoundError(
                f"Training dataset not found: {self.input_csv}"
            )
        if self.min_train_days <= 0:
            raise ValueError("min_train_days must be positive.")
        if self.test_window_days <= 0:
            raise ValueError("test_window_days must be positive.")


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Overall or per-window multiclass evaluation metrics."""

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


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    """Return the repository root from this ml script."""
    return Path(__file__).resolve().parents[1]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the version 3 training dataset."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Training dataset is empty: {path}")

    required = {
        "match_card_id",
        "match_date",
        TARGET_COLUMN,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Training dataset missing required columns: {missing}"
        )

    duplicate_ids = int(frame["match_card_id"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(
            f"Training dataset contains {duplicate_ids} duplicate match IDs."
        )

    frame = frame.copy()
    frame["match_date"] = pd.to_datetime(
        frame["match_date"],
        errors="coerce",
    )
    frame[TARGET_COLUMN] = (
        frame[TARGET_COLUMN].astype(str).str.strip().str.upper()
    )

    frame = frame.loc[
        frame["match_date"].notna()
        & frame[TARGET_COLUMN].isin(CLASS_ORDER)
    ].copy()
    frame = frame.sort_values(
        ["match_date", "match_card_id"],
        kind="stable",
    ).reset_index(drop=True)

    if frame.empty:
        raise ValueError(
            "No valid dated A/D/H rows remain after validation."
        )

    LOGGER.info(
        "Loaded dataset: rows=%d columns=%d dates=%s to %s",
        len(frame),
        len(frame.columns),
        frame["match_date"].min().date(),
        frame["match_date"].max().date(),
    )
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Select all nonconstant numeric model features."""
    numeric = frame.select_dtypes(include="number").columns.tolist()
    candidates = [
        column
        for column in numeric
        if column not in EXCLUDED_COLUMNS
    ]

    if not candidates:
        raise ValueError("No numeric model features were found.")

    constant = [
        column
        for column in candidates
        if frame[column].nunique(dropna=True) <= 1
    ]
    selected = [
        column
        for column in candidates
        if column not in constant
    ]

    h2h_features = [
        column for column in selected if column.startswith("h2h_")
    ]
    if not h2h_features:
        raise ValueError(
            "No H2H features were selected. "
            "Confirm that training_dataset_v3.csv is being used."
        )

    LOGGER.info(
        "Numeric features: %d | Constant removed: %d | "
        "Selected: %d | H2H: %d",
        len(candidates),
        len(constant),
        len(selected),
        len(h2h_features),
    )
    return selected


def align_probabilities(
    probabilities: np.ndarray,
    model_classes: list[str],
) -> np.ndarray:
    """Align estimator probability columns to A, D, H order."""
    aligned = np.zeros(
        (len(probabilities), len(CLASS_ORDER)),
        dtype=float,
    )

    for source_index, label in enumerate(model_classes):
        if label in CLASS_ORDER:
            target_index = CLASS_ORDER.index(label)
            aligned[:, target_index] = probabilities[:, source_index]

    row_sums = aligned.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze() <= 0
    aligned[~zero_rows] = (
        aligned[~zero_rows] / row_sums[~zero_rows]
    )
    aligned[zero_rows] = 1.0 / len(CLASS_ORDER)
    return aligned


def evaluate_predictions(
    actual: pd.Series,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> EvaluationMetrics:
    """Calculate multiclass metrics in canonical class order."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=CLASS_ORDER,
        zero_division=0,
    )

    return EvaluationMetrics(
        rows=len(actual),
        accuracy=float(accuracy_score(actual, predicted)),
        macro_f1=float(
            f1_score(actual, predicted, average="macro")
        ),
        weighted_f1=float(
            f1_score(actual, predicted, average="weighted")
        ),
        log_loss=float(
            log_loss(actual, probabilities, labels=CLASS_ORDER)
        ),
        away_precision=float(precision[0]),
        away_recall=float(recall[0]),
        away_f1=float(f1[0]),
        draw_precision=float(precision[1]),
        draw_recall=float(recall[1]),
        draw_f1=float(f1[1]),
        home_precision=float(precision[2]),
        home_recall=float(recall[2]),
        home_f1=float(f1[2]),
    )


def build_model(config: BacktestConfig) -> Pipeline:
    """Build the same RandomForest specification used by v5."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=700,
                    max_depth=14,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=config.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def rolling_backtest(
    frame: pd.DataFrame,
    features: list[str],
    config: BacktestConfig,
    model_factory: Callable[[], Pipeline],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run expanding-window backtests with strict date separation."""
    first_date = frame["match_date"].min()
    last_date = frame["match_date"].max()
    test_start = first_date + pd.Timedelta(
        days=config.min_train_days
    )

    prediction_parts: list[pd.DataFrame] = []
    windows: list[dict[str, object]] = []
    importance_parts: list[pd.DataFrame] = []
    window_id = 0

    while test_start <= last_date:
        test_end = min(
            test_start
            + pd.Timedelta(days=config.test_window_days - 1),
            last_date,
        )

        train = frame.loc[
            frame["match_date"] < test_start
        ].copy()
        test = frame.loc[
            (frame["match_date"] >= test_start)
            & (frame["match_date"] <= test_end)
        ].copy()

        if not train.empty and not test.empty:
            if train["match_date"].max() >= test_start:
                raise RuntimeError(
                    "Future leakage detected in training window."
                )

            window_id += 1
            model = model_factory()
            model.fit(train[features], train[TARGET_COLUMN])

            predicted = model.predict(test[features])
            raw_probability = model.predict_proba(test[features])
            classes = list(model.named_steps["model"].classes_)
            probabilities = align_probabilities(
                raw_probability,
                classes,
            )
            metrics = evaluate_predictions(
                test[TARGET_COLUMN],
                predicted,
                probabilities,
            )

            prediction_parts.append(
                pd.DataFrame(
                    {
                        "window": window_id,
                        "match_card_id": test[
                            "match_card_id"
                        ].values,
                        "match_date": test[
                            "match_date"
                        ].dt.strftime("%Y-%m-%d").values,
                        "season": (
                            test["season"].values
                            if "season" in test.columns
                            else test["match_date"].dt.year.values
                        ),
                        "competition": (
                            test["competition"].values
                            if "competition" in test.columns
                            else "J.League"
                        ),
                        "home_team": (
                            test["home_team"].values
                            if "home_team" in test.columns
                            else ""
                        ),
                        "away_team": (
                            test["away_team"].values
                            if "away_team" in test.columns
                            else ""
                        ),
                        "actual": test[TARGET_COLUMN].values,
                        "prediction": predicted,
                        "prob_away": probabilities[:, 0],
                        "prob_draw": probabilities[:, 1],
                        "prob_home": probabilities[:, 2],
                    }
                )
            )

            window_record = {
                "window": window_id,
                "train_from": train[
                    "match_date"
                ].min().strftime("%Y-%m-%d"),
                "train_to": train[
                    "match_date"
                ].max().strftime("%Y-%m-%d"),
                "test_from": test_start.strftime("%Y-%m-%d"),
                "test_to": test_end.strftime("%Y-%m-%d"),
                "train_rows": len(train),
                "test_rows": len(test),
                **asdict(metrics),
            }
            windows.append(window_record)

            estimator = model.named_steps["model"]
            importance_values = estimator.feature_importances_
            importance_total = float(
                np.sum(importance_values)
            )
            normalized = (
                importance_values / importance_total
                if importance_total > 0
                else importance_values
            )
            importance_parts.append(
                pd.DataFrame(
                    {
                        "window": window_id,
                        "feature": features,
                        "importance": normalized,
                    }
                )
            )

            LOGGER.info(
                "Window %02d | Train %s-%s (%d) | "
                "Test %s-%s (%d) | Accuracy %.4f | "
                "MacroF1 %.4f | DrawRecall %.4f",
                window_id,
                window_record["train_from"],
                window_record["train_to"],
                len(train),
                window_record["test_from"],
                window_record["test_to"],
                len(test),
                metrics.accuracy,
                metrics.macro_f1,
                metrics.draw_recall,
            )

        test_start = test_end + pd.Timedelta(days=1)

    if not prediction_parts:
        raise RuntimeError(
            "No backtest windows were produced."
        )

    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.DataFrame(windows),
        pd.concat(importance_parts, ignore_index=True),
    )


def save_outputs(
    config: BacktestConfig,
    predictions: pd.DataFrame,
    windows: pd.DataFrame,
    importances: pd.DataFrame,
    overall: EvaluationMetrics,
) -> pd.DataFrame:
    """Save predictions, metrics and feature importance."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    importance_summary = (
        importances.groupby(
            "feature",
            as_index=False,
        )["importance"]
        .mean()
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    predictions.to_csv(
        config.output_dir / "randomforest_v6_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    windows.to_csv(
        config.output_dir / "randomforest_v6_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    importance_summary.to_csv(
        config.output_dir
        / "randomforest_v6_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "model": "RandomForest v6",
        **asdict(overall),
    }
    pd.DataFrame([summary]).to_csv(
        config.output_dir / "randomforest_v6_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        config.output_dir / "randomforest_v6_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return importance_summary


def train_final_model(
    frame: pd.DataFrame,
    features: list[str],
    config: BacktestConfig,
) -> Path:
    """Fit and persist a final model using every valid row."""
    model = build_model(config)
    model.fit(frame[features], frame[TARGET_COLUMN])

    model_path = (
        config.output_dir / "random_forest_v6.joblib"
    )
    joblib.dump(
        {
            "model": model,
            "features": features,
            "class_order": CLASS_ORDER,
            "trained_through": frame[
                "match_date"
            ].max().strftime("%Y-%m-%d"),
            "training_dataset": str(config.input_csv),
        },
        model_path,
    )
    return model_path


def print_results(
    frame: pd.DataFrame,
    features: list[str],
    predictions: pd.DataFrame,
    overall: EvaluationMetrics,
    importance: pd.DataFrame,
    config: BacktestConfig,
    model_path: Path,
) -> None:
    """Print the final backtest summary."""
    h2h_features = [
        feature
        for feature in features
        if feature.startswith("h2h_")
    ]

    print("=" * 72)
    print("Project Alpha RandomForest v6")
    print("=" * 72)
    print(
        f"Dataset              : "
        f"{frame['match_date'].min().date()} to "
        f"{frame['match_date'].max().date()}"
    )
    print(f"Dataset rows         : {len(frame)}")
    print(f"Backtest rows        : {len(predictions)}")
    print(f"Features             : {len(features)}")
    print(f"H2H features         : {len(h2h_features)}")
    print(f"Accuracy             : {overall.accuracy:.6f}")
    print(f"Macro F1             : {overall.macro_f1:.6f}")
    print(f"Weighted F1          : {overall.weighted_f1:.6f}")
    print(f"Log Loss             : {overall.log_loss:.6f}")
    print(f"Away F1              : {overall.away_f1:.6f}")
    print(f"Draw F1              : {overall.draw_f1:.6f}")
    print(f"Home F1              : {overall.home_f1:.6f}")
    print()
    print("Classification Report")
    print("-" * 72)
    print(
        classification_report(
            predictions["actual"],
            predictions["prediction"],
            labels=CLASS_ORDER,
            zero_division=0,
        )
    )
    print("Confusion Matrix [A, D, H]")
    print(
        confusion_matrix(
            predictions["actual"],
            predictions["prediction"],
            labels=CLASS_ORDER,
        )
    )
    print()
    print("Top 30 Features")
    print("-" * 72)
    print(importance.head(30).to_string(index=False))
    print()
    print(f"Model saved          : {model_path}")
    print(f"Outputs              : {config.output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Project Alpha RandomForest v6 "
            "date-based H2H backtest."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "ml" / "training_dataset_v3.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "ml" / "rf_v6",
    )
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=365,
    )
    parser.add_argument(
        "--test-window-days",
        type=int,
        default=90,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    configure_logging(args.verbose)

    config = BacktestConfig(
        input_csv=args.input,
        output_dir=args.output_dir,
        min_train_days=args.min_train_days,
        test_window_days=args.test_window_days,
        random_state=args.random_state,
    )
    config.validate()

    frame = load_dataset(config.input_csv)
    features = feature_columns(frame)

    predictions, windows, importances = rolling_backtest(
        frame=frame,
        features=features,
        config=config,
        model_factory=lambda: build_model(config),
    )
    overall = evaluate_predictions(
        predictions["actual"],
        predictions["prediction"].to_numpy(),
        predictions[
            ["prob_away", "prob_draw", "prob_home"]
        ].to_numpy(),
    )
    importance_summary = save_outputs(
        config=config,
        predictions=predictions,
        windows=windows,
        importances=importances,
        overall=overall,
    )
    model_path = train_final_model(
        frame=frame,
        features=features,
        config=config,
    )
    print_results(
        frame=frame,
        features=features,
        predictions=predictions,
        overall=overall,
        importance=importance_summary,
        config=config,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
