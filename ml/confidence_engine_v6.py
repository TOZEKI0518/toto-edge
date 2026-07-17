from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("confidence_engine_v6")

OUTCOMES = ("A", "D", "H")
PROBABILITY_COLUMNS = {
    "A": "prob_away",
    "D": "prob_draw",
    "H": "prob_home",
}
RF_PROBABILITY_COLUMNS = {
    "A": "rf_prob_away",
    "D": "rf_prob_draw",
    "H": "rf_prob_home",
}
LGBM_PROBABILITY_COLUMNS = {
    "A": "lgbm_prob_away",
    "D": "lgbm_prob_draw",
    "H": "lgbm_prob_home",
}
MARKET_PROBABILITY_COLUMNS = {
    "A": "market_prob_away",
    "D": "market_prob_draw",
    "H": "market_prob_home",
}


@dataclass(frozen=True, slots=True)
class ConfidenceConfig:
    """Configuration for confidence and coverage classification."""

    input_predictions_csv: Path
    output_dir: Path
    market_csv: Path | None = None
    single_threshold: float = 0.48
    double_threshold: float = 0.40
    single_margin_threshold: float = 0.08
    double_margin_threshold: float = 0.03
    max_single_disagreement: float = 0.14
    max_double_disagreement: float = 0.20
    high_confidence_threshold: float = 0.45
    medium_confidence_threshold: float = 0.30

    def validate(self) -> None:
        if not self.input_predictions_csv.exists():
            raise FileNotFoundError(
                f"Prediction CSV was not found: {self.input_predictions_csv}"
            )
        for name, value in (
            ("single_threshold", self.single_threshold),
            ("double_threshold", self.double_threshold),
            ("single_margin_threshold", self.single_margin_threshold),
            ("double_margin_threshold", self.double_margin_threshold),
            ("max_single_disagreement", self.max_single_disagreement),
            ("max_double_disagreement", self.max_double_disagreement),
            ("high_confidence_threshold", self.high_confidence_threshold),
            ("medium_confidence_threshold", self.medium_confidence_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if self.double_threshold > self.single_threshold:
            raise ValueError(
                "double_threshold must not exceed single_threshold."
            )
        if self.market_csv is not None and not self.market_csv.exists():
            raise FileNotFoundError(
                f"Market CSV was not found: {self.market_csv}"
            )


@dataclass(frozen=True, slots=True)
class ConfidenceSummary:
    """Summary of confidence-engine output."""

    rows: int
    high_confidence_rows: int
    medium_confidence_rows: int
    low_confidence_rows: int
    single_rows: int
    double_rows: int
    triple_rows: int
    overall_accuracy: float | None
    single_accuracy: float | None
    double_coverage_rate: float | None
    triple_coverage_rate: float | None
    mean_confidence_score: float
    mean_model_disagreement: float
    market_data_available: bool


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    """Return repository root from this script under ml/."""
    return Path(__file__).resolve().parents[1]


def normalize_probabilities(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Normalize probability columns and reject invalid values."""
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(
            f"Probability columns contain missing/non-numeric values: {columns}"
        )
    if (values < 0).any().any():
        raise ValueError("Probability columns contain negative values.")

    row_sums = values.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("At least one probability row sums to zero.")

    return values.div(row_sums, axis=0)


def validate_predictions(frame: pd.DataFrame) -> None:
    """Validate required identifiers and probability schemas."""
    required = {
        "match_card_id",
        *PROBABILITY_COLUMNS.values(),
        *RF_PROBABILITY_COLUMNS.values(),
        *LGBM_PROBABILITY_COLUMNS.values(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Prediction CSV is missing required columns: "
            + ", ".join(missing)
        )

    duplicate_ids = int(frame["match_card_id"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(
            f"Prediction CSV contains {duplicate_ids} duplicate match IDs."
        )


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return entropy normalized to the range 0..1."""
    clipped = np.clip(probabilities, 1e-15, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return entropy / math.log(probabilities.shape[1])


def jensen_shannon_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Return normalized Jensen-Shannon distance per row."""
    midpoint = 0.5 * (first + second)
    first_clipped = np.clip(first, 1e-15, 1.0)
    second_clipped = np.clip(second, 1e-15, 1.0)
    midpoint_clipped = np.clip(midpoint, 1e-15, 1.0)

    kl_first = np.sum(
        first_clipped * np.log(first_clipped / midpoint_clipped),
        axis=1,
    )
    kl_second = np.sum(
        second_clipped * np.log(second_clipped / midpoint_clipped),
        axis=1,
    )
    divergence = 0.5 * (kl_first + kl_second)
    return np.sqrt(np.clip(divergence / math.log(2.0), 0.0, 1.0))


def classify_confidence(
    score: pd.Series,
    config: ConfidenceConfig,
) -> pd.Series:
    """Map numeric confidence to high/medium/low labels."""
    return pd.Series(
        np.select(
            [
                score >= config.high_confidence_threshold,
                score >= config.medium_confidence_threshold,
            ],
            ["HIGH", "MEDIUM"],
            default="LOW",
        ),
        index=score.index,
        dtype="string",
    )


def choose_coverage(
    top_probability: pd.Series,
    probability_margin: pd.Series,
    disagreement: pd.Series,
    config: ConfidenceConfig,
) -> pd.Series:
    """Recommend single, double or triple coverage."""
    single = (
        (top_probability >= config.single_threshold)
        & (probability_margin >= config.single_margin_threshold)
        & (disagreement <= config.max_single_disagreement)
    )
    double = (
        (top_probability >= config.double_threshold)
        & (probability_margin >= config.double_margin_threshold)
        & (disagreement <= config.max_double_disagreement)
    )

    return pd.Series(
        np.select(
            [single, double],
            ["SINGLE", "DOUBLE"],
            default="TRIPLE",
        ),
        index=top_probability.index,
        dtype="string",
    )


def recommendation_from_probabilities(
    probabilities: np.ndarray,
    coverage: pd.Series,
) -> tuple[list[str], list[str], list[str]]:
    """Create first, second and combined outcome recommendations."""
    order = np.argsort(-probabilities, axis=1)
    first = [OUTCOMES[index] for index in order[:, 0]]
    second = [OUTCOMES[index] for index in order[:, 1]]

    combinations: list[str] = []
    for row_index, coverage_type in enumerate(coverage):
        if coverage_type == "SINGLE":
            combinations.append(first[row_index])
        elif coverage_type == "DOUBLE":
            combinations.append(
                "".join(
                    OUTCOMES[index]
                    for index in order[row_index, :2]
                )
            )
        else:
            combinations.append("ADH")

    return first, second, combinations


def attach_market_data(
    frame: pd.DataFrame,
    market_csv: Path | None,
) -> tuple[pd.DataFrame, bool]:
    """Optionally merge market/public-pick probabilities."""
    if market_csv is None:
        return frame, False

    market = pd.read_csv(market_csv)
    required = {
        "match_card_id",
        *MARKET_PROBABILITY_COLUMNS.values(),
    }
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(
            "Market CSV is missing required columns: "
            + ", ".join(missing)
        )

    if market["match_card_id"].duplicated().any():
        raise ValueError("Market CSV contains duplicate match_card_id values.")

    normalized = normalize_probabilities(
        market,
        list(MARKET_PROBABILITY_COLUMNS.values()),
    )
    market = market[["match_card_id"]].copy()
    for column in MARKET_PROBABILITY_COLUMNS.values():
        market[column] = normalized[column]

    merged = frame.merge(
        market,
        on="match_card_id",
        how="left",
        validate="one_to_one",
    )

    missing_market = int(
        merged[list(MARKET_PROBABILITY_COLUMNS.values())]
        .isna()
        .any(axis=1)
        .sum()
    )
    if missing_market:
        LOGGER.warning(
            "Market data missing for %d prediction rows.",
            missing_market,
        )

    return merged, True


def build_confidence_features(
    predictions: pd.DataFrame,
    config: ConfidenceConfig,
) -> pd.DataFrame:
    """Create confidence, uncertainty and coverage features."""
    validate_predictions(predictions)
    frame = predictions.copy()

    ensemble_columns = list(PROBABILITY_COLUMNS.values())
    rf_columns = list(RF_PROBABILITY_COLUMNS.values())
    lgbm_columns = list(LGBM_PROBABILITY_COLUMNS.values())

    ensemble = normalize_probabilities(frame, ensemble_columns)
    rf = normalize_probabilities(frame, rf_columns)
    lgbm = normalize_probabilities(frame, lgbm_columns)

    frame[ensemble_columns] = ensemble
    frame[rf_columns] = rf
    frame[lgbm_columns] = lgbm

    probability_values = ensemble.to_numpy()
    rf_values = rf.to_numpy()
    lgbm_values = lgbm.to_numpy()

    sorted_probabilities = np.sort(probability_values, axis=1)[:, ::-1]
    frame["top_probability"] = sorted_probabilities[:, 0]
    frame["second_probability"] = sorted_probabilities[:, 1]
    frame["third_probability"] = sorted_probabilities[:, 2]
    frame["probability_margin"] = (
        frame["top_probability"] - frame["second_probability"]
    )
    frame["probability_entropy"] = normalized_entropy(
        probability_values
    )
    frame["certainty_from_entropy"] = 1.0 - frame["probability_entropy"]

    frame["model_disagreement_js"] = jensen_shannon_distance(
        rf_values,
        lgbm_values,
    )
    frame["model_max_probability_gap"] = np.max(
        np.abs(rf_values - lgbm_values),
        axis=1,
    )
    rf_top = np.argmax(rf_values, axis=1)
    lgbm_top = np.argmax(lgbm_values, axis=1)
    frame["models_agree_top_pick"] = rf_top == lgbm_top

    agreement_component = 1.0 - frame["model_disagreement_js"]
    margin_component = np.clip(
        frame["probability_margin"] / 0.35,
        0.0,
        1.0,
    )
    top_probability_component = np.clip(
        (frame["top_probability"] - (1.0 / 3.0)) / (2.0 / 3.0),
        0.0,
        1.0,
    )

    frame["confidence_score"] = (
        0.35 * frame["certainty_from_entropy"]
        + 0.30 * margin_component
        + 0.20 * agreement_component
        + 0.15 * top_probability_component
    ).clip(0.0, 1.0)

    frame["confidence_level"] = classify_confidence(
        frame["confidence_score"],
        config,
    )
    frame["coverage_recommendation"] = choose_coverage(
        frame["top_probability"],
        frame["probability_margin"],
        frame["model_disagreement_js"],
        config,
    )

    first, second, combination = recommendation_from_probabilities(
        probability_values,
        frame["coverage_recommendation"],
    )
    frame["primary_pick"] = first
    frame["secondary_pick"] = second
    frame["recommended_combination"] = combination

    frame["uncertainty_score"] = (
        1.0 - frame["confidence_score"]
    )
    frame["draw_risk_score"] = (
        frame["prob_draw"]
        * (1.0 - frame["probability_margin"])
        * (1.0 + frame["model_disagreement_js"])
    ).clip(0.0, 1.0)

    frame, market_available = attach_market_data(
        frame,
        config.market_csv,
    )

    if market_available:
        for outcome in OUTCOMES:
            model_column = PROBABILITY_COLUMNS[outcome]
            market_column = MARKET_PROBABILITY_COLUMNS[outcome]
            suffix = {
                "A": "away",
                "D": "draw",
                "H": "home",
            }[outcome]

            frame[f"edge_{suffix}"] = (
                frame[model_column] - frame[market_column]
            )
            frame[f"value_ratio_{suffix}"] = np.where(
                frame[market_column] > 0,
                frame[model_column] / frame[market_column],
                np.nan,
            )

        edge_columns = ["edge_away", "edge_draw", "edge_home"]
        value_columns = [
            "value_ratio_away",
            "value_ratio_draw",
            "value_ratio_home",
        ]
        frame["best_edge"] = frame[edge_columns].max(axis=1)
        frame["best_value_ratio"] = frame[value_columns].max(axis=1)
        edge_index = frame[edge_columns].to_numpy().argmax(axis=1)
        frame["best_value_pick"] = [
            OUTCOMES[index] for index in edge_index
        ]
        frame["roi_priority_score"] = (
            frame["confidence_score"]
            * np.clip(frame["best_edge"], 0.0, None)
            * np.clip(frame["best_value_ratio"], 0.0, 3.0)
        )
    else:
        frame["roi_priority_score"] = np.nan

    return frame


def calculate_coverage_accuracy(
    frame: pd.DataFrame,
) -> dict[str, float | None]:
    """Calculate historical hit/coverage metrics when actual results exist."""
    if "actual" not in frame.columns:
        return {
            "overall_accuracy": None,
            "single_accuracy": None,
            "double_coverage_rate": None,
            "triple_coverage_rate": None,
        }

    actual = frame["actual"].astype(str).str.upper()
    primary_correct = frame["primary_pick"] == actual

    single_mask = frame["coverage_recommendation"] == "SINGLE"
    double_mask = frame["coverage_recommendation"] == "DOUBLE"
    triple_mask = frame["coverage_recommendation"] == "TRIPLE"

    recommended = frame["recommended_combination"].astype(str)
    double_covered = pd.Series(
        [
            result in combination
            for result, combination in zip(
                actual,
                recommended,
                strict=False,
            )
        ],
        index=frame.index,
        dtype=bool,
    )
    triple_covered = double_covered.copy()

    def safe_mean(mask: pd.Series, values: pd.Series) -> float | None:
        if int(mask.sum()) == 0:
            return None
        return float(values.loc[mask].mean())

    return {
        "overall_accuracy": float(primary_correct.mean()),
        "single_accuracy": safe_mean(single_mask, primary_correct),
        "double_coverage_rate": safe_mean(double_mask, double_covered),
        "triple_coverage_rate": safe_mean(triple_mask, triple_covered),
    }


def build_summary(
    frame: pd.DataFrame,
    market_available: bool,
) -> ConfidenceSummary:
    """Build aggregate diagnostics."""
    metrics = calculate_coverage_accuracy(frame)

    return ConfidenceSummary(
        rows=len(frame),
        high_confidence_rows=int(
            (frame["confidence_level"] == "HIGH").sum()
        ),
        medium_confidence_rows=int(
            (frame["confidence_level"] == "MEDIUM").sum()
        ),
        low_confidence_rows=int(
            (frame["confidence_level"] == "LOW").sum()
        ),
        single_rows=int(
            (frame["coverage_recommendation"] == "SINGLE").sum()
        ),
        double_rows=int(
            (frame["coverage_recommendation"] == "DOUBLE").sum()
        ),
        triple_rows=int(
            (frame["coverage_recommendation"] == "TRIPLE").sum()
        ),
        overall_accuracy=metrics["overall_accuracy"],
        single_accuracy=metrics["single_accuracy"],
        double_coverage_rate=metrics["double_coverage_rate"],
        triple_coverage_rate=metrics["triple_coverage_rate"],
        mean_confidence_score=float(
            frame["confidence_score"].mean()
        ),
        mean_model_disagreement=float(
            frame["model_disagreement_js"].mean()
        ),
        market_data_available=market_available,
    )


def save_outputs(
    frame: pd.DataFrame,
    summary: ConfidenceSummary,
    config: ConfidenceConfig,
) -> None:
    """Save confidence rows, diagnostics and summaries."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(
        config.output_dir / "confidence_v6_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confidence_distribution = (
        frame.groupby(
            ["confidence_level", "coverage_recommendation"],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
    )
    confidence_distribution.to_csv(
        config.output_dir / "confidence_v6_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if "actual" in frame.columns:
        diagnostics = (
            frame.assign(
                primary_correct=(
                    frame["primary_pick"]
                    == frame["actual"].astype(str).str.upper()
                )
            )
            .groupby(
                ["confidence_level", "coverage_recommendation"],
                dropna=False,
            )
            .agg(
                rows=("match_card_id", "size"),
                primary_accuracy=("primary_correct", "mean"),
                mean_top_probability=("top_probability", "mean"),
                mean_confidence=("confidence_score", "mean"),
                mean_disagreement=("model_disagreement_js", "mean"),
            )
            .reset_index()
        )
        diagnostics.to_csv(
            config.output_dir / "confidence_v6_backtest.csv",
            index=False,
            encoding="utf-8-sig",
        )

    payload: dict[str, Any] = {
        **asdict(summary),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    (
        config.output_dir / "confidence_v6_summary.json"
    ).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([asdict(summary)]).to_csv(
        config.output_dir / "confidence_v6_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def print_summary(
    summary: ConfidenceSummary,
    config: ConfidenceConfig,
) -> None:
    """Print concise execution results."""
    def format_metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.6f}"

    print("=" * 78)
    print("Project Alpha Confidence Engine v6")
    print("=" * 78)
    print(f"Rows                    : {summary.rows}")
    print(f"High confidence         : {summary.high_confidence_rows}")
    print(f"Medium confidence       : {summary.medium_confidence_rows}")
    print(f"Low confidence          : {summary.low_confidence_rows}")
    print(f"Single recommendation   : {summary.single_rows}")
    print(f"Double recommendation   : {summary.double_rows}")
    print(f"Triple recommendation   : {summary.triple_rows}")
    print(f"Primary accuracy        : {format_metric(summary.overall_accuracy)}")
    print(f"Single accuracy         : {format_metric(summary.single_accuracy)}")
    print(
        f"Double coverage rate    : "
        f"{format_metric(summary.double_coverage_rate)}"
    )
    print(
        f"Triple coverage rate    : "
        f"{format_metric(summary.triple_coverage_rate)}"
    )
    print(
        f"Mean confidence         : "
        f"{summary.mean_confidence_score:.6f}"
    )
    print(
        f"Mean disagreement       : "
        f"{summary.mean_model_disagreement:.6f}"
    )
    print(
        f"Market data available   : "
        f"{summary.market_data_available}"
    )
    print(f"Outputs                 : {config.output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Generate confidence, uncertainty and coverage signals "
            "from Ensemble v6 predictions."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            root
            / "ml"
            / "ensemble_v6"
            / "ensemble_v6_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "ml" / "confidence_v6",
    )
    parser.add_argument(
        "--market-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing match_card_id and "
            "market_prob_away/draw/home."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    configure_logging(args.verbose)

    config = ConfidenceConfig(
        input_predictions_csv=args.input,
        output_dir=args.output_dir,
        market_csv=args.market_csv,
    )
    config.validate()

    predictions = pd.read_csv(config.input_predictions_csv)
    confidence = build_confidence_features(
        predictions=predictions,
        config=config,
    )
    market_available = all(
        column in confidence.columns
        for column in MARKET_PROBABILITY_COLUMNS.values()
    )
    summary = build_summary(
        frame=confidence,
        market_available=market_available,
    )
    save_outputs(
        frame=confidence,
        summary=summary,
        config=config,
    )
    print_summary(summary, config)


if __name__ == "__main__":
    main()
