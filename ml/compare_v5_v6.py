from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("compare_v5_v6")

METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "log_loss",
    "away_f1",
    "draw_f1",
    "home_f1",
]

DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "log_loss": "Log Loss",
    "away_f1": "Away F1",
    "draw_f1": "Draw F1",
    "home_f1": "Home F1",
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Description of one model result source."""

    model: str
    version: str
    directory: Path
    summary_stems: tuple[str, ...]
    prediction_stems: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.model}_{self.version}".lower()


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_specs(root: Path) -> list[ModelSpec]:
    ml_root = root / "ml"
    return [
        ModelSpec(
            model="RandomForest",
            version="v5",
            directory=ml_root / "rf_v5",
            summary_stems=(
                "randomforest_v5_summary",
                "random_forest_v5_summary",
                "summary",
                "metrics",
            ),
            prediction_stems=(
                "randomforest_v5_predictions",
                "random_forest_v5_predictions",
                "predictions",
            ),
        ),
        ModelSpec(
            model="RandomForest",
            version="v6",
            directory=ml_root / "rf_v6",
            summary_stems=(
                "randomforest_v6_summary",
                "random_forest_v6_summary",
                "summary",
                "metrics",
            ),
            prediction_stems=(
                "randomforest_v6_predictions",
                "random_forest_v6_predictions",
                "predictions",
            ),
        ),
        ModelSpec(
            model="LightGBM",
            version="v5",
            directory=ml_root / "lgbm_v5",
            summary_stems=(
                "lgbm_v5_summary",
                "lightgbm_v5_summary",
                "summary",
                "metrics",
            ),
            prediction_stems=(
                "lgbm_v5_predictions",
                "lightgbm_v5_predictions",
                "predictions",
            ),
        ),
        ModelSpec(
            model="LightGBM",
            version="v6",
            directory=ml_root / "lgbm_v6",
            summary_stems=(
                "lgbm_v6_summary",
                "lightgbm_v6_summary",
                "summary",
                "metrics",
            ),
            prediction_stems=(
                "lgbm_v6_predictions",
                "lightgbm_v6_predictions",
                "predictions",
            ),
        ),
    ]


def normalize_key(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def normalize_mapping(data: dict[str, Any]) -> dict[str, Any]:
    normalized = {normalize_key(str(key)): value for key, value in data.items()}

    aliases = {
        "accuracy_score": "accuracy",
        "macro_f1_score": "macro_f1",
        "f1_macro": "macro_f1",
        "weighted_f1_score": "weighted_f1",
        "f1_weighted": "weighted_f1",
        "logloss": "log_loss",
        "away_f1_score": "away_f1",
        "draw_f1_score": "draw_f1",
        "home_f1_score": "home_f1",
    }
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]

    return normalized


def find_candidate(
    directory: Path,
    stems: tuple[str, ...],
    suffixes: tuple[str, ...],
) -> Path | None:
    for stem in stems:
        for suffix in suffixes:
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

    if directory.exists():
        for suffix in suffixes:
            matches = sorted(directory.glob(f"*{suffix}"))
            if matches:
                preferred = [
                    path for path in matches
                    if any(stem in path.stem.lower() for stem in stems)
                ]
                if preferred:
                    return preferred[0]
    return None


def read_summary(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError(f"Summary JSON must contain an object: {path}")
        return normalize_mapping(payload)

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Summary CSV is empty: {path}")

    if len(frame) == 1:
        return normalize_mapping(frame.iloc[0].to_dict())

    if frame.shape[1] >= 2:
        first = frame.columns[0]
        second = frame.columns[1]
        mapping = dict(zip(frame[first], frame[second], strict=False))
        return normalize_mapping(mapping)

    raise ValueError(f"Unsupported summary CSV format: {path}")


def compute_from_predictions(path: Path) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        precision_recall_fscore_support,
    )

    frame = pd.read_csv(path)
    required = {"actual", "prediction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Prediction file missing columns {missing}: {path}"
        )

    actual = frame["actual"].astype(str).str.upper()
    predicted = frame["prediction"].astype(str).str.upper()
    labels = ["A", "D", "H"]

    _, _, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=labels,
        zero_division=0,
    )

    result = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(actual, predicted, average="macro")
        ),
        "weighted_f1": float(
            f1_score(actual, predicted, average="weighted")
        ),
        "away_f1": float(f1[0]),
        "draw_f1": float(f1[1]),
        "home_f1": float(f1[2]),
    }

    probability_columns = ["prob_away", "prob_draw", "prob_home"]
    if all(column in frame.columns for column in probability_columns):
        result["log_loss"] = float(
            log_loss(
                actual,
                frame[probability_columns].to_numpy(),
                labels=labels,
            )
        )

    return result


def load_model_metrics(spec: ModelSpec) -> tuple[dict[str, float], str]:
    summary_path = find_candidate(
        spec.directory,
        spec.summary_stems,
        (".csv", ".json"),
    )
    values: dict[str, Any] = {}
    source_parts: list[str] = []

    if summary_path is not None:
        values.update(read_summary(summary_path))
        source_parts.append(summary_path.name)

    missing_metrics = [
        metric for metric in METRICS
        if metric not in values or pd.isna(values[metric])
    ]

    if missing_metrics:
        predictions_path = find_candidate(
            spec.directory,
            spec.prediction_stems,
            (".csv",),
        )
        if predictions_path is not None:
            prediction_metrics = compute_from_predictions(predictions_path)
            for metric, value in prediction_metrics.items():
                values.setdefault(metric, value)
            source_parts.append(predictions_path.name)

    still_missing = [
        metric for metric in METRICS
        if metric not in values or pd.isna(values[metric])
    ]
    if still_missing:
        raise FileNotFoundError(
            f"Could not load metrics {still_missing} for "
            f"{spec.model} {spec.version} from {spec.directory}"
        )

    result = {
        metric: float(values[metric])
        for metric in METRICS
    }
    return result, ", ".join(source_parts)


def build_comparison(specs: list[ModelSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        metrics, source = load_model_metrics(spec)
        rows.append(
            {
                "model": spec.model,
                "version": spec.version,
                "label": f"{spec.model} {spec.version}",
                **metrics,
                "source": source,
            }
        )
        LOGGER.info(
            "Loaded %s %s: Accuracy=%.6f MacroF1=%.6f LogLoss=%.6f",
            spec.model,
            spec.version,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["log_loss"],
        )

    return pd.DataFrame(rows)


def add_version_deltas(comparison: pd.DataFrame) -> pd.DataFrame:
    result = comparison.copy()

    for model in result["model"].unique():
        subset = result.loc[result["model"] == model].set_index("version")
        if "v5" not in subset.index or "v6" not in subset.index:
            continue

        for metric in METRICS:
            delta = float(
                subset.loc["v6", metric] - subset.loc["v5", metric]
            )
            mask = (
                (result["model"] == model)
                & (result["version"] == "v6")
            )
            result.loc[mask, f"{metric}_vs_v5"] = delta

    return result


def build_delta_table(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in comparison["model"].unique():
        subset = comparison.loc[
            comparison["model"] == model
        ].set_index("version")
        if "v5" not in subset.index or "v6" not in subset.index:
            continue

        row: dict[str, Any] = {"model": model}
        for metric in METRICS:
            v5_value = float(subset.loc["v5", metric])
            v6_value = float(subset.loc["v6", metric])
            delta = v6_value - v5_value
            improved = delta < 0 if metric == "log_loss" else delta > 0

            row[f"{metric}_v5"] = v5_value
            row[f"{metric}_v6"] = v6_value
            row[f"{metric}_delta"] = delta
            row[f"{metric}_improved"] = improved
        rows.append(row)

    return pd.DataFrame(rows)


def select_best_models(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        ascending = metric == "log_loss"
        ordered = comparison.sort_values(
            metric,
            ascending=ascending,
        )
        best = ordered.iloc[0]
        rows.append(
            {
                "metric": DISPLAY_NAMES[metric],
                "best_model": best["label"],
                "best_value": float(best[metric]),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    comparison: pd.DataFrame,
    deltas: pd.DataFrame,
    best_models: pd.DataFrame,
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )
    deltas.to_csv(
        output_csv.with_name("model_comparison_v6_deltas.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    best_models.to_csv(
        output_csv.with_name("model_comparison_v6_best_models.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "models": comparison.to_dict(orient="records"),
        "deltas": deltas.to_dict(orient="records"),
        "best_models": best_models.to_dict(orient="records"),
    }
    output_csv.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(
    comparison: pd.DataFrame,
    deltas: pd.DataFrame,
    best_models: pd.DataFrame,
    output_csv: Path,
) -> None:
    columns = [
        "label",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "away_f1",
        "draw_f1",
        "home_f1",
    ]

    print("=" * 110)
    print("Project Alpha Model Comparison: v5 vs v6")
    print("=" * 110)
    print(
        comparison[columns]
        .rename(
            columns={
                "label": "Model",
                **DISPLAY_NAMES,
            }
        )
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )

    print()
    print("Version 6 change versus Version 5")
    print("-" * 110)
    for _, row in deltas.iterrows():
        print(row["model"])
        for metric in METRICS:
            delta = float(row[f"{metric}_delta"])
            direction = "改善" if bool(
                row[f"{metric}_improved"]
            ) else "悪化"
            print(
                f"  {DISPLAY_NAMES[metric]:<12}: "
                f"{delta:+.6f} ({direction})"
            )

    print()
    print("Best model by metric")
    print("-" * 110)
    print(
        best_models.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"Saved: {output_csv}")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Compare Project Alpha RF/LGBM v5 and v6 metrics."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "ml" / "model_comparison_v6.csv",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    specs = default_specs(project_root())
    comparison = build_comparison(specs)
    comparison = add_version_deltas(comparison)
    deltas = build_delta_table(comparison)
    best_models = select_best_models(comparison)

    save_outputs(
        comparison=comparison,
        deltas=deltas,
        best_models=best_models,
        output_csv=args.output,
    )
    print_summary(
        comparison=comparison,
        deltas=deltas,
        best_models=best_models,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
