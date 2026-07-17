from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("current_round_inference_v6")

CLASS_ORDER = ("A", "D", "H")
PROBABILITY_COLUMNS = {
    "A": "prob_away",
    "D": "prob_draw",
    "H": "prob_home",
}


class CurrentRoundInferenceError(RuntimeError):
    """Raised when current-round inference cannot be completed safely."""


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def load_model_bundle(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise CurrentRoundInferenceError(
            f"{label} must contain a dictionary bundle."
        )

    required = {"model", "features"}
    missing = sorted(required - set(bundle))
    if missing:
        raise CurrentRoundInferenceError(
            f"{label} is missing bundle keys: {missing}"
        )

    features = bundle["features"]
    if not isinstance(features, (list, tuple)) or not features:
        raise CurrentRoundInferenceError(
            f"{label} contains an invalid feature list."
        )

    LOGGER.info(
        "Loaded %s: features=%d trained_through=%s",
        label,
        len(features),
        bundle.get("trained_through", "unknown"),
    )
    return bundle


def load_current_features(path: Path) -> pd.DataFrame:
    require_file(path, "current-round feature CSV")
    frame = pd.read_csv(path)

    if len(frame) != 13:
        raise CurrentRoundInferenceError(
            f"Expected 13 current-round rows, found {len(frame)}."
        )

    required = {
        "match_card_id",
        "match_date",
        "home_team",
        "away_team",
        "round_id",
        "toto_match_no",
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CurrentRoundInferenceError(
            "Current-round feature CSV is missing columns: "
            + ", ".join(missing)
        )

    duplicate_ids = int(frame["match_card_id"].duplicated().sum())
    if duplicate_ids:
        raise CurrentRoundInferenceError(
            f"Current-round feature CSV contains "
            f"{duplicate_ids} duplicate match IDs."
        )

    frame = frame.sort_values(
        "toto_match_no",
        kind="stable",
    ).reset_index(drop=True)

    expected_numbers = list(range(1, 14))
    actual_numbers = frame["toto_match_no"].astype(int).tolist()
    if actual_numbers != expected_numbers:
        raise CurrentRoundInferenceError(
            f"Expected toto_match_no 1..13, found {actual_numbers}."
        )

    LOGGER.info(
        "Loaded current-round features: rows=%d columns=%d",
        len(frame),
        len(frame.columns),
    )
    return frame


def align_feature_matrix(
    frame: pd.DataFrame,
    features: list[str],
    model_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    aligned = pd.DataFrame(index=frame.index)
    missing_columns: list[str] = []
    non_numeric_columns: list[str] = []

    for column in features:
        if column not in frame.columns:
            aligned[column] = np.nan
            missing_columns.append(column)
            continue

        converted = pd.to_numeric(
            frame[column],
            errors="coerce",
        )
        if converted.isna().all() and frame[column].notna().any():
            non_numeric_columns.append(column)
        aligned[column] = converted

    infinite_count = int(
        np.isinf(
            aligned.to_numpy(dtype=float, copy=False)
        ).sum()
    )
    if infinite_count:
        aligned = aligned.replace([np.inf, -np.inf], np.nan)

    diagnostics = {
        "model": model_label,
        "feature_count": len(features),
        "missing_feature_columns": missing_columns,
        "missing_feature_column_count": len(missing_columns),
        "non_numeric_feature_columns": non_numeric_columns,
        "non_numeric_feature_column_count": len(non_numeric_columns),
        "missing_values_before_pipeline": int(
            aligned.isna().sum().sum()
        ),
        "infinite_values_replaced": infinite_count,
    }

    LOGGER.info(
        "%s feature alignment: features=%d missing_columns=%d "
        "missing_values=%d",
        model_label,
        len(features),
        len(missing_columns),
        diagnostics["missing_values_before_pipeline"],
    )
    return aligned, diagnostics


def align_probabilities(
    probabilities: np.ndarray,
    model_classes: list[str],
) -> np.ndarray:
    aligned = np.zeros(
        (len(probabilities), len(CLASS_ORDER)),
        dtype=float,
    )

    for source_index, label in enumerate(model_classes):
        normalized = str(label).strip().upper()
        if normalized in CLASS_ORDER:
            target_index = CLASS_ORDER.index(normalized)
            aligned[:, target_index] = probabilities[:, source_index]

    totals = aligned.sum(axis=1, keepdims=True)
    invalid = totals.squeeze() <= 0
    aligned[~invalid] = aligned[~invalid] / totals[~invalid]
    aligned[invalid] = 1.0 / len(CLASS_ORDER)
    return aligned


def predict_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    label: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model = bundle["model"]
    features = list(bundle["features"])
    matrix, diagnostics = align_feature_matrix(
        frame=frame,
        features=features,
        model_label=label,
    )

    if not hasattr(model, "predict_proba"):
        raise CurrentRoundInferenceError(
            f"{label} model does not support predict_proba()."
        )

    raw_probabilities = model.predict_proba(matrix)

    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model")
        model_classes = list(
            getattr(estimator, "classes_", CLASS_ORDER)
        )
    else:
        model_classes = list(
            getattr(model, "classes_", CLASS_ORDER)
        )

    probabilities = align_probabilities(
        probabilities=np.asarray(raw_probabilities, dtype=float),
        model_classes=model_classes,
    )
    predictions = np.array(
        [
            CLASS_ORDER[index]
            for index in probabilities.argmax(axis=1)
        ],
        dtype=object,
    )

    diagnostics["probability_sum_min"] = float(
        probabilities.sum(axis=1).min()
    )
    diagnostics["probability_sum_max"] = float(
        probabilities.sum(axis=1).max()
    )
    return predictions, probabilities, diagnostics


def build_output(
    frame: pd.DataFrame,
    rf_predictions: np.ndarray,
    rf_probabilities: np.ndarray,
    lgbm_predictions: np.ndarray,
    lgbm_probabilities: np.ndarray,
    rf_weight: float,
) -> pd.DataFrame:
    lgbm_weight = 1.0 - rf_weight
    ensemble = (
        rf_weight * rf_probabilities
        + lgbm_weight * lgbm_probabilities
    )
    ensemble = ensemble / ensemble.sum(axis=1, keepdims=True)
    ensemble_predictions = np.array(
        [
            CLASS_ORDER[index]
            for index in ensemble.argmax(axis=1)
        ],
        dtype=object,
    )

    output_columns = [
        "round_id",
        "toto_match_no",
        "match_card_id",
        "match_date",
        "home_team",
        "away_team",
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
    ]
    output = frame[output_columns].copy()

    output["rf_prediction"] = rf_predictions
    output["rf_prob_away"] = rf_probabilities[:, 0]
    output["rf_prob_draw"] = rf_probabilities[:, 1]
    output["rf_prob_home"] = rf_probabilities[:, 2]

    output["lgbm_prediction"] = lgbm_predictions
    output["lgbm_prob_away"] = lgbm_probabilities[:, 0]
    output["lgbm_prob_draw"] = lgbm_probabilities[:, 1]
    output["lgbm_prob_home"] = lgbm_probabilities[:, 2]

    output["prediction"] = ensemble_predictions
    output[PROBABILITY_COLUMNS["A"]] = ensemble[:, 0]
    output[PROBABILITY_COLUMNS["D"]] = ensemble[:, 1]
    output[PROBABILITY_COLUMNS["H"]] = ensemble[:, 2]

    output["rf_weight"] = rf_weight
    output["lgbm_weight"] = lgbm_weight

    output["top_probability"] = ensemble.max(axis=1)
    sorted_probabilities = np.sort(ensemble, axis=1)[:, ::-1]
    output["second_probability"] = sorted_probabilities[:, 1]
    output["probability_margin"] = (
        sorted_probabilities[:, 0]
        - sorted_probabilities[:, 1]
    )

    output["models_agree"] = (
        rf_predictions == lgbm_predictions
    )
    output["model_max_probability_gap"] = np.max(
        np.abs(rf_probabilities - lgbm_probabilities),
        axis=1,
    )

    output["edge_away"] = (
        output["prob_away"] - output["market_prob_away"]
    )
    output["edge_draw"] = (
        output["prob_draw"] - output["market_prob_draw"]
    )
    output["edge_home"] = (
        output["prob_home"] - output["market_prob_home"]
    )

    edge_values = output[
        ["edge_away", "edge_draw", "edge_home"]
    ].to_numpy()
    output["best_edge"] = edge_values.max(axis=1)
    output["best_edge_pick"] = [
        CLASS_ORDER[index]
        for index in edge_values.argmax(axis=1)
    ]

    return output


def build_summary(
    output: pd.DataFrame,
    rf_diagnostics: dict[str, Any],
    lgbm_diagnostics: dict[str, Any],
    rf_weight: float,
) -> dict[str, Any]:
    return {
        "rows": int(len(output)),
        "round_ids": sorted(
            output["round_id"].astype(int).unique().tolist()
        ),
        "rf_weight": rf_weight,
        "lgbm_weight": 1.0 - rf_weight,
        "duplicate_match_card_ids": int(
            output["match_card_id"].duplicated().sum()
        ),
        "missing_output_values": int(
            output.isna().sum().sum()
        ),
        "rf_lgbm_agreement_rows": int(
            output["models_agree"].sum()
        ),
        "prediction_counts": {
            label: int((output["prediction"] == label).sum())
            for label in CLASS_ORDER
        },
        "mean_top_probability": float(
            output["top_probability"].mean()
        ),
        "mean_probability_margin": float(
            output["probability_margin"].mean()
        ),
        "mean_model_probability_gap": float(
            output["model_max_probability_gap"].mean()
        ),
        "mean_best_edge": float(
            output["best_edge"].mean()
        ),
        "rf_diagnostics": rf_diagnostics,
        "lgbm_diagnostics": lgbm_diagnostics,
    }


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run RandomForest v6, LightGBM v6 and RF60/LGBM40 "
            "ensemble inference for the current toto round."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            root
            / "ml"
            / "current_round"
            / "current_round_features_v6.csv"
        ),
    )
    parser.add_argument(
        "--rf-model",
        type=Path,
        default=root / "ml" / "rf_v6" / "random_forest_v6.joblib",
    )
    parser.add_argument(
        "--lgbm-model",
        type=Path,
        default=root / "ml" / "lgbm_v6" / "lightgbm_v6.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "ml"
            / "current_round"
            / "current_round_predictions_v6.csv"
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=(
            root
            / "ml"
            / "diagnostics"
            / "current_round"
            / "current_round_inference_v6_summary.json"
        ),
    )
    parser.add_argument(
        "--rf-weight",
        type=float,
        default=0.60,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if not 0.0 <= args.rf_weight <= 1.0:
        raise ValueError("--rf-weight must be between 0 and 1.")

    frame = load_current_features(args.input)
    rf_bundle = load_model_bundle(
        args.rf_model,
        "RandomForest v6 model",
    )
    lgbm_bundle = load_model_bundle(
        args.lgbm_model,
        "LightGBM v6 model",
    )

    rf_predictions, rf_probabilities, rf_diagnostics = (
        predict_bundle(
            bundle=rf_bundle,
            frame=frame,
            label="RandomForest v6",
        )
    )
    lgbm_predictions, lgbm_probabilities, lgbm_diagnostics = (
        predict_bundle(
            bundle=lgbm_bundle,
            frame=frame,
            label="LightGBM v6",
        )
    )

    output = build_output(
        frame=frame,
        rf_predictions=rf_predictions,
        rf_probabilities=rf_probabilities,
        lgbm_predictions=lgbm_predictions,
        lgbm_probabilities=lgbm_probabilities,
        rf_weight=args.rf_weight,
    )

    summary = build_summary(
        output=output,
        rf_diagnostics=rf_diagnostics,
        lgbm_diagnostics=lgbm_diagnostics,
        rf_weight=args.rf_weight,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )
    args.summary_json.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 92)
    print("Project Alpha Current Round Inference v6")
    print("=" * 92)
    print(f"Rows                       : {summary['rows']}")
    print(
        f"Duplicate match IDs        : "
        f"{summary['duplicate_match_card_ids']}"
    )
    print(
        f"Missing output values      : "
        f"{summary['missing_output_values']}"
    )
    print(
        f"RF/LGBM agreement rows     : "
        f"{summary['rf_lgbm_agreement_rows']}"
    )
    print(
        f"Prediction counts          : "
        f"{summary['prediction_counts']}"
    )
    print(
        f"Mean top probability       : "
        f"{summary['mean_top_probability']:.6f}"
    )
    print(
        f"Mean probability margin    : "
        f"{summary['mean_probability_margin']:.6f}"
    )
    print(
        f"Mean best edge             : "
        f"{summary['mean_best_edge']:.6f}"
    )
    print(f"Output                     : {args.output}")
    print(f"Diagnostics                : {args.summary_json}")


if __name__ == "__main__":
    main()
