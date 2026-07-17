from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline


LOGGER = logging.getLogger("feature_ablation_v5")
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
class Config:
    training_csv: Path
    low_importance_csv: Path
    output_dir: Path
    min_train_days: int = 365
    test_window_days: int = 90
    random_state: int = 42


@dataclass(frozen=True)
class Metrics:
    feature_count: int
    removed_count: int
    rows: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    log_loss: float


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_training(path: Path) -> pd.DataFrame:
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
        raise ValueError("No valid dated H/D/A rows remain.")

    return frame


def load_removal_candidates(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Low-importance feature file not found: {path}"
        )

    frame = pd.read_csv(path)
    if "feature" not in frame.columns:
        raise ValueError(
            f"Low-importance file missing 'feature' column: {path}"
        )

    if "consensus_score" in frame.columns:
        frame = frame.sort_values("consensus_score", ascending=True)
    elif "consensus_rank" in frame.columns:
        frame = frame.sort_values("consensus_rank", ascending=False)

    return frame["feature"].astype(str).drop_duplicates().tolist()


def all_numeric_features(frame: pd.DataFrame) -> list[str]:
    candidates = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in EXCLUDED_COLUMNS
    ]

    return [
        column
        for column in candidates
        if frame[column].nunique(dropna=True) > 1
    ]


def build_model(config: Config) -> Pipeline:
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


def rolling_predictions(
    frame: pd.DataFrame,
    features: list[str],
    config: Config,
    model_factory: Callable[[], Pipeline],
) -> pd.DataFrame:
    first_date = frame["match_date"].min()
    last_date = frame["match_date"].max()
    test_start = first_date + pd.Timedelta(days=config.min_train_days)

    outputs: list[pd.DataFrame] = []
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

            prediction = model.predict(test[features])
            raw_probability = model.predict_proba(test[features])
            classes = list(model.named_steps["model"].classes_)
            probability = align_probabilities(raw_probability, classes)

            outputs.append(
                pd.DataFrame(
                    {
                        "window": window_id,
                        "match_card_id": test["match_card_id"].values,
                        "match_date": test["match_date"].dt.strftime(
                            "%Y-%m-%d"
                        ).values,
                        "actual": test[TARGET_COLUMN].values,
                        "prediction": prediction,
                        "prob_away": probability[:, 0],
                        "prob_draw": probability[:, 1],
                        "prob_home": probability[:, 2],
                    }
                )
            )

        test_start = test_end + pd.Timedelta(days=1)

    if not outputs:
        raise RuntimeError("No rolling-backtest windows were produced.")

    return pd.concat(outputs, ignore_index=True)


def evaluate(
    predictions: pd.DataFrame,
    feature_count: int,
    removed_count: int,
) -> Metrics:
    actual = predictions["actual"]
    predicted = predictions["prediction"]
    probability = predictions[
        ["prob_away", "prob_draw", "prob_home"]
    ].to_numpy()

    return Metrics(
        feature_count=feature_count,
        removed_count=removed_count,
        rows=len(predictions),
        accuracy=float(accuracy_score(actual, predicted)),
        macro_f1=float(f1_score(actual, predicted, average="macro")),
        weighted_f1=float(
            f1_score(actual, predicted, average="weighted")
        ),
        log_loss=float(log_loss(actual, probability, labels=CLASS_ORDER)),
    )


def candidate_removal_counts(candidate_count: int) -> list[int]:
    return sorted(
        {
            min(value, candidate_count)
            for value in [0, 5, 10, 15, 20]
        }
    )


def run(config: Config) -> None:
    frame = load_training(config.training_csv)
    baseline_features = all_numeric_features(frame)
    removal_candidates = load_removal_candidates(
        config.low_importance_csv
    )
    removal_candidates = [
        feature
        for feature in removal_candidates
        if feature in baseline_features
    ]

    LOGGER.info("Training rows: %s", len(frame))
    LOGGER.info("Baseline features: %s", len(baseline_features))
    LOGGER.info(
        "Valid removal candidates: %s",
        len(removal_candidates),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    experiment_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    selected_sets: dict[int, list[str]] = {}

    for removed_count in candidate_removal_counts(
        len(removal_candidates)
    ):
        removed = set(removal_candidates[:removed_count])
        features = [
            feature
            for feature in baseline_features
            if feature not in removed
        ]

        LOGGER.info(
            "Experiment remove=%s features=%s",
            removed_count,
            len(features),
        )

        predictions = rolling_predictions(
            frame,
            features,
            config,
            lambda: build_model(config),
        )
        metrics = evaluate(
            predictions,
            feature_count=len(features),
            removed_count=removed_count,
        )

        experiment_rows.append(
            {
                **asdict(metrics),
                "removed_features": "|".join(
                    removal_candidates[:removed_count]
                ),
            }
        )

        predictions.insert(0, "removed_count", removed_count)
        prediction_frames.append(predictions)
        selected_sets[removed_count] = features

        LOGGER.info(
            "remove=%02d Accuracy=%.4f MacroF1=%.4f "
            "WeightedF1=%.4f LogLoss=%.4f",
            removed_count,
            metrics.accuracy,
            metrics.macro_f1,
            metrics.weighted_f1,
            metrics.log_loss,
        )

    results = pd.DataFrame(experiment_rows).sort_values(
        ["accuracy", "macro_f1", "log_loss"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_removed_count = int(results.iloc[0]["removed_count"])
    best_features = selected_sets[best_removed_count]

    results.to_csv(
        config.output_dir / "feature_ablation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        config.output_dir / "feature_ablation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "feature": best_features,
            "selected": True,
        }
    ).to_csv(
        config.output_dir / "selected_features_v5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (config.output_dir / "selected_features_v5.json").write_text(
        json.dumps(best_features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_summary = {
        "best_removed_count": best_removed_count,
        "best_feature_count": len(best_features),
        "best_accuracy": float(results.iloc[0]["accuracy"]),
        "best_macro_f1": float(results.iloc[0]["macro_f1"]),
        "best_weighted_f1": float(results.iloc[0]["weighted_f1"]),
        "best_log_loss": float(results.iloc[0]["log_loss"]),
        "removed_features": removal_candidates[:best_removed_count],
    }

    (config.output_dir / "feature_ablation_summary.json").write_text(
        json.dumps(best_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("Project Alpha Feature Ablation v5")
    print("=" * 72)
    print(results.to_string(index=False))
    print()
    print(f"Best removed count : {best_removed_count}")
    print(f"Best feature count : {len(best_features)}")
    print(f"Best accuracy      : {best_summary['best_accuracy']:.4f}")
    print(f"Best macro F1      : {best_summary['best_macro_f1']:.4f}")
    print(f"Best log loss      : {best_summary['best_log_loss']:.4f}")
    print(f"Saved              : {config.output_dir}")


def parse_args() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RandomForest v5 after removing low-importance features."
        )
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=root / "ml" / "training_dataset_v2.csv",
    )
    parser.add_argument(
        "--low-importance",
        type=Path,
        default=(
            root
            / "ml"
            / "feature_importance_analysis"
            / "low_importance_candidates.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "ml" / "feature_ablation_v5",
    )
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--test-window-days", type=int, default=90)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    config = Config(
        training_csv=args.training,
        low_importance_csv=args.low_importance,
        output_dir=args.output_dir,
        min_train_days=args.min_train_days,
        test_window_days=args.test_window_days,
        random_state=args.random_state,
    )
    run(config)


if __name__ == "__main__":
    main()
