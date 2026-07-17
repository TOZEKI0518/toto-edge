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

LOGGER = logging.getLogger("random_forest_v5")
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


@dataclass(frozen=True)
class BacktestConfig:
    input_csv: Path
    output_dir: Path
    min_train_days: int = 365
    test_window_days: int = 90
    random_state: int = 42


@dataclass(frozen=True)
class EvaluationMetrics:
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
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Training dataset is empty: {path}")

    required = {"match_date", TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training dataset missing columns: {sorted(missing)}")

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
        raise ValueError("No valid dated H/D/A rows remain after validation.")

    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    features = [
        column for column in numeric if column not in EXCLUDED_COLUMNS
    ]

    if not features:
        raise ValueError("No numeric model features were found.")

    constant = [
        column
        for column in features
        if frame[column].nunique(dropna=True) <= 1
    ]
    selected = [column for column in features if column not in constant]

    LOGGER.info(
        "Numeric features: %s | Constant removed: %s | Selected: %s",
        len(features),
        len(constant),
        len(selected),
    )
    return selected


def align_probabilities(
    probabilities: np.ndarray,
    model_classes: list[str],
) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(CLASS_ORDER)), dtype=float)
    for source_index, label in enumerate(model_classes):
        if label in CLASS_ORDER:
            target_index = CLASS_ORDER.index(label)
            aligned[:, target_index] = probabilities[:, source_index]

    row_sums = aligned.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze() <= 0
    aligned[~zero_rows] = aligned[~zero_rows] / row_sums[~zero_rows]
    aligned[zero_rows] = 1.0 / len(CLASS_ORDER)
    return aligned


def evaluate_predictions(
    actual: pd.Series,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> EvaluationMetrics:
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=CLASS_ORDER,
        zero_division=0,
    )

    return EvaluationMetrics(
        rows=len(actual),
        accuracy=float(accuracy_score(actual, predicted)),
        macro_f1=float(f1_score(actual, predicted, average="macro")),
        weighted_f1=float(f1_score(actual, predicted, average="weighted")),
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
    )


def rolling_backtest(
    frame: pd.DataFrame,
    features: list[str],
    config: BacktestConfig,
    model_factory: Callable[[], Pipeline],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first_date = frame["match_date"].min()
    last_date = frame["match_date"].max()
    test_start = first_date + pd.Timedelta(days=config.min_train_days)

    predictions: list[pd.DataFrame] = []
    windows: list[dict[str, object]] = []
    importances: list[pd.DataFrame] = []
    window_id = 0

    while test_start <= last_date:
        test_end = min(
            test_start + pd.Timedelta(days=config.test_window_days - 1),
            last_date,
        )

        train = frame.loc[frame["match_date"] < test_start]
        test = frame.loc[
            (frame["match_date"] >= test_start)
            & (frame["match_date"] <= test_end)
        ]

        if len(train) > 0 and len(test) > 0:
            window_id += 1
            model = model_factory()
            model.fit(train[features], train[TARGET_COLUMN])

            predicted = model.predict(test[features])
            raw_probability = model.predict_proba(test[features])
            classes = list(model.named_steps["model"].classes_)
            probabilities = align_probabilities(raw_probability, classes)
            metrics = evaluate_predictions(
                test[TARGET_COLUMN],
                predicted,
                probabilities,
            )

            prediction_frame = pd.DataFrame(
                {
                    "window": window_id,
                    "match_card_id": test["match_card_id"].values,
                    "match_date": test["match_date"].dt.strftime(
                        "%Y-%m-%d"
                    ).values,
                    "season": test["season"].values,
                    "competition": test["competition"].values,
                    "home_team": test["home_team"].values,
                    "away_team": test["away_team"].values,
                    "actual": test[TARGET_COLUMN].values,
                    "prediction": predicted,
                    "prob_away": probabilities[:, 0],
                    "prob_draw": probabilities[:, 1],
                    "prob_home": probabilities[:, 2],
                }
            )
            predictions.append(prediction_frame)

            windows.append(
                {
                    "window": window_id,
                    "train_from": train["match_date"].min().strftime(
                        "%Y-%m-%d"
                    ),
                    "train_to": train["match_date"].max().strftime(
                        "%Y-%m-%d"
                    ),
                    "test_from": test_start.strftime("%Y-%m-%d"),
                    "test_to": test_end.strftime("%Y-%m-%d"),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    **asdict(metrics),
                }
            )

            estimator = model.named_steps["model"]
            importance_values = getattr(
                estimator,
                "feature_importances_",
                np.zeros(len(features), dtype=float),
            )
            total = float(np.sum(importance_values))
            normalized = (
                importance_values / total if total > 0 else importance_values
            )
            importances.append(
                pd.DataFrame(
                    {
                        "window": window_id,
                        "feature": features,
                        "importance": normalized,
                    }
                )
            )

            LOGGER.info(
                "Window %02d | Train %s-%s (%s) | Test %s-%s (%s) | "
                "Accuracy %.4f | MacroF1 %.4f | DrawRecall %.4f",
                window_id,
                windows[-1]["train_from"],
                windows[-1]["train_to"],
                len(train),
                windows[-1]["test_from"],
                windows[-1]["test_to"],
                len(test),
                metrics.accuracy,
                metrics.macro_f1,
                metrics.draw_recall,
            )

        test_start = test_end + pd.Timedelta(days=1)

    if not predictions:
        raise RuntimeError(
            "No backtest windows were produced. "
            "Reduce --min-train-days or --test-window-days."
        )

    prediction_df = pd.concat(predictions, ignore_index=True)
    window_df = pd.DataFrame(windows)
    importance_df = pd.concat(importances, ignore_index=True)
    return prediction_df, window_df, importance_df


def save_common_outputs(
    output_dir: Path,
    model_name: str,
    predictions: pd.DataFrame,
    windows: pd.DataFrame,
    importance: pd.DataFrame,
    overall: EvaluationMetrics,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = model_name.lower().replace(" ", "_")
    predictions.to_csv(
        output_dir / f"{prefix}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    windows.to_csv(
        output_dir / f"{prefix}_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    importance_summary = (
        importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    importance_summary.to_csv(
        output_dir / f"{prefix}_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame([{"model": model_name, **asdict(overall)}])
    summary.to_csv(
        output_dir / f"{prefix}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(
            {"model": model_name, **asdict(overall)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return importance_summary


def print_results(
    model_name: str,
    frame: pd.DataFrame,
    features: list[str],
    predictions: pd.DataFrame,
    overall: EvaluationMetrics,
    importance: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("=" * 72)
    print(model_name)
    print("=" * 72)
    print(
        f"Dataset              : {frame['match_date'].min().date()} "
        f"to {frame['match_date'].max().date()}"
    )
    print(f"Dataset rows         : {len(frame)}")
    print(f"Backtest rows        : {len(predictions)}")
    print(f"Features             : {len(features)}")
    print(f"Accuracy             : {overall.accuracy:.4f}")
    print(f"Macro F1             : {overall.macro_f1:.4f}")
    print(f"Weighted F1          : {overall.weighted_f1:.4f}")
    print(f"Log Loss             : {overall.log_loss:.4f}")
    print(f"Draw Precision       : {overall.draw_precision:.4f}")
    print(f"Draw Recall          : {overall.draw_recall:.4f}")
    print(f"Draw F1              : {overall.draw_f1:.4f}")
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
    print(f"Saved               : {output_dir}")


from sklearn.ensemble import RandomForestClassifier


def build_model(config: BacktestConfig) -> Pipeline:
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


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Project Alpha RandomForest v5 date-based backtest."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "ml" / "training_dataset_v2.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "ml" / "rf_v5",
    )
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--test-window-days", type=int, default=90)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    config = BacktestConfig(
        input_csv=args.input,
        output_dir=args.output_dir,
        min_train_days=args.min_train_days,
        test_window_days=args.test_window_days,
        random_state=args.random_state,
    )
    frame = load_dataset(config.input_csv)
    features = feature_columns(frame)

    predictions, windows, importance = rolling_backtest(
        frame,
        features,
        config,
        lambda: build_model(config),
    )
    overall = evaluate_predictions(
        predictions["actual"],
        predictions["prediction"].to_numpy(),
        predictions[["prob_away", "prob_draw", "prob_home"]].to_numpy(),
    )
    importance_summary = save_common_outputs(
        config.output_dir,
        "RandomForest v5",
        predictions,
        windows,
        importance,
        overall,
    )

    final_model = build_model(config)
    final_model.fit(frame[features], frame[TARGET_COLUMN])
    joblib.dump(
        {
            "model": final_model,
            "features": features,
            "class_order": CLASS_ORDER,
            "trained_through": frame["match_date"].max().strftime(
                "%Y-%m-%d"
            ),
        },
        config.output_dir / "random_forest_v5.joblib",
    )

    print_results(
        "Project Alpha RandomForest v5",
        frame,
        features,
        predictions,
        overall,
        importance_summary,
        config.output_dir,
    )


if __name__ == "__main__":
    main()
