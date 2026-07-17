from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("market_prediction_merger_v6")

JOIN_COLUMNS = ("round_id", "toto_match_no")
AI_PROBABILITY_COLUMNS = {
    "A": "prob_away",
    "D": "prob_draw",
    "H": "prob_home",
}
MARKET_PROBABILITY_COLUMNS = {
    "A": "market_prob_away",
    "D": "market_prob_draw",
    "H": "market_prob_home",
}
OUTCOME_LABELS = ("A", "D", "H")


class MarketPredictionMergeError(RuntimeError):
    """Raised when market and prediction data cannot be merged safely."""


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


def load_csv(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)
    frame = pd.read_csv(path)
    if frame.empty:
        raise MarketPredictionMergeError(f"{label} is empty: {path}")
    LOGGER.info(
        "Loaded %s: rows=%d columns=%d",
        label,
        len(frame),
        len(frame.columns),
    )
    return frame


def validate_unique_key(
    frame: pd.DataFrame,
    label: str,
) -> None:
    missing = sorted(set(JOIN_COLUMNS) - set(frame.columns))
    if missing:
        raise MarketPredictionMergeError(
            f"{label} is missing join columns: {missing}"
        )

    if frame[list(JOIN_COLUMNS)].isna().any().any():
        raise MarketPredictionMergeError(
            f"{label} contains missing join-key values."
        )

    duplicates = int(
        frame.duplicated(subset=list(JOIN_COLUMNS)).sum()
    )
    if duplicates:
        raise MarketPredictionMergeError(
            f"{label} contains {duplicates} duplicate join keys."
        )


def normalize_probability_group(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> pd.DataFrame:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")

    if values.isna().any().any():
        raise MarketPredictionMergeError(
            f"{label} probabilities contain missing/non-numeric values."
        )
    if (values < 0).any().any():
        raise MarketPredictionMergeError(
            f"{label} probabilities contain negative values."
        )

    totals = values.sum(axis=1)
    if (totals <= 0).any():
        raise MarketPredictionMergeError(
            f"{label} contains zero-sum probability rows."
        )

    return values.div(totals, axis=0)


def prepare_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        *JOIN_COLUMNS,
        "match_card_id",
        "match_date",
        "home_team",
        "away_team",
        "prediction",
        *AI_PROBABILITY_COLUMNS.values(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MarketPredictionMergeError(
            "Prediction CSV is missing required columns: "
            + ", ".join(missing)
        )

    validate_unique_key(frame, "prediction CSV")

    output = frame.copy()
    ai_columns = list(AI_PROBABILITY_COLUMNS.values())
    output[ai_columns] = normalize_probability_group(
        output,
        ai_columns,
        "AI",
    )

    output["prediction"] = (
        output["prediction"].astype(str).str.strip().str.upper()
    )
    invalid_predictions = sorted(
        set(output["prediction"]) - set(OUTCOME_LABELS)
    )
    if invalid_predictions:
        raise MarketPredictionMergeError(
            f"Invalid prediction labels: {invalid_predictions}"
        )

    return output


def prepare_market(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        *JOIN_COLUMNS,
        "home_team",
        "away_team",
        *MARKET_PROBABILITY_COLUMNS.values(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MarketPredictionMergeError(
            "Market CSV is missing required columns: "
            + ", ".join(missing)
        )

    validate_unique_key(frame, "market CSV")

    output = frame.copy()
    market_columns = list(MARKET_PROBABILITY_COLUMNS.values())
    output[market_columns] = normalize_probability_group(
        output,
        market_columns,
        "Market",
    )

    output = output.rename(
        columns={
            "home_team": "market_home_team",
            "away_team": "market_away_team",
        }
    )
    return output


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    return np.where(
        denominator > 0,
        numerator / denominator,
        np.nan,
    )


def add_market_intelligence_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    for outcome in OUTCOME_LABELS:
        ai_column = AI_PROBABILITY_COLUMNS[outcome]
        market_column = MARKET_PROBABILITY_COLUMNS[outcome]
        suffix = {
            "A": "away",
            "D": "draw",
            "H": "home",
        }[outcome]

        output[f"edge_{suffix}"] = (
            output[ai_column] - output[market_column]
        )
        output[f"value_ratio_{suffix}"] = safe_divide(
            output[ai_column],
            output[market_column],
        )
        output[f"log_value_ratio_{suffix}"] = np.log(
            output[f"value_ratio_{suffix}"].clip(lower=1e-12)
        )

    edge_columns = ["edge_away", "edge_draw", "edge_home"]
    value_ratio_columns = [
        "value_ratio_away",
        "value_ratio_draw",
        "value_ratio_home",
    ]

    edge_values = output[edge_columns].to_numpy(dtype=float)
    value_values = output[value_ratio_columns].to_numpy(dtype=float)

    output["best_edge"] = np.nanmax(edge_values, axis=1)
    output["best_edge_pick"] = [
        OUTCOME_LABELS[index]
        for index in np.nanargmax(edge_values, axis=1)
    ]

    output["best_value_ratio"] = np.nanmax(value_values, axis=1)
    output["best_value_pick"] = [
        OUTCOME_LABELS[index]
        for index in np.nanargmax(value_values, axis=1)
    ]

    output["ai_top_probability"] = output[
        list(AI_PROBABILITY_COLUMNS.values())
    ].max(axis=1)
    output["market_top_probability"] = output[
        list(MARKET_PROBABILITY_COLUMNS.values())
    ].max(axis=1)

    ai_order = np.argsort(
        -output[list(AI_PROBABILITY_COLUMNS.values())].to_numpy(),
        axis=1,
    )
    market_order = np.argsort(
        -output[list(MARKET_PROBABILITY_COLUMNS.values())].to_numpy(),
        axis=1,
    )

    output["ai_top_pick"] = [
        OUTCOME_LABELS[index] for index in ai_order[:, 0]
    ]
    output["market_top_pick"] = [
        OUTCOME_LABELS[index] for index in market_order[:, 0]
    ]
    output["ai_market_agree"] = (
        output["ai_top_pick"] == output["market_top_pick"]
    )

    ai_sorted = np.sort(
        output[list(AI_PROBABILITY_COLUMNS.values())].to_numpy(),
        axis=1,
    )[:, ::-1]
    market_sorted = np.sort(
        output[list(MARKET_PROBABILITY_COLUMNS.values())].to_numpy(),
        axis=1,
    )[:, ::-1]

    output["ai_probability_margin"] = (
        ai_sorted[:, 0] - ai_sorted[:, 1]
    )
    output["market_probability_margin"] = (
        market_sorted[:, 0] - market_sorted[:, 1]
    )

    output["market_overreaction_score"] = (
        output["market_top_probability"]
        - output["ai_top_probability"]
    ).abs()

    output["value_score"] = (
        output["best_edge"].clip(lower=0.0)
        * output["best_value_ratio"].clip(lower=0.0, upper=5.0)
        * (
            0.5
            + 0.5
            * output["ai_top_probability"].clip(lower=0.0, upper=1.0)
        )
    )

    output["roi_priority_score"] = (
        output["value_score"]
        * (
            1.0
            + output["ai_probability_margin"].clip(
                lower=0.0,
                upper=1.0,
            )
        )
    )

    output["is_positive_edge"] = output["best_edge"] > 0
    output["is_strong_edge"] = output["best_edge"] >= 0.10
    output["is_extreme_edge"] = output["best_edge"] >= 0.20

    return output


def verify_team_alignment(frame: pd.DataFrame) -> None:
    home_mismatch = frame["home_team"].astype(str).str.strip().ne(
        frame["market_home_team"].astype(str).str.strip()
    )
    away_mismatch = frame["away_team"].astype(str).str.strip().ne(
        frame["market_away_team"].astype(str).str.strip()
    )

    mismatch = home_mismatch | away_mismatch
    if mismatch.any():
        examples = frame.loc[
            mismatch,
            [
                "round_id",
                "toto_match_no",
                "home_team",
                "away_team",
                "market_home_team",
                "market_away_team",
            ],
        ].head(10)

        raise MarketPredictionMergeError(
            "Team-name mismatch detected between prediction and market "
            "data:\n"
            + examples.to_string(index=False)
        )


def merge_market_and_predictions(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    merged = predictions.merge(
        market,
        on=list(JOIN_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator="_merge_status",
        suffixes=("", "_market"),
    )

    unmatched = merged["_merge_status"].ne("both")
    if unmatched.any():
        examples = merged.loc[
            unmatched,
            list(JOIN_COLUMNS),
        ].head(10)
        raise MarketPredictionMergeError(
            "Prediction rows without matching market data:\n"
            + examples.to_string(index=False)
        )

    merged = merged.drop(columns="_merge_status")
    verify_team_alignment(merged)
    merged = add_market_intelligence_features(merged)

    return merged.sort_values(
        list(JOIN_COLUMNS),
        kind="stable",
    ).reset_index(drop=True)


def build_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "round_ids": sorted(
            frame["round_id"].astype(int).unique().tolist()
        ),
        "duplicate_join_keys": int(
            frame.duplicated(subset=list(JOIN_COLUMNS)).sum()
        ),
        "missing_values": int(frame.isna().sum().sum()),
        "positive_edge_rows": int(frame["is_positive_edge"].sum()),
        "strong_edge_rows": int(frame["is_strong_edge"].sum()),
        "extreme_edge_rows": int(frame["is_extreme_edge"].sum()),
        "ai_market_agreement_rows": int(
            frame["ai_market_agree"].sum()
        ),
        "mean_best_edge": float(frame["best_edge"].mean()),
        "max_best_edge": float(frame["best_edge"].max()),
        "mean_best_value_ratio": float(
            frame["best_value_ratio"].mean()
        ),
        "mean_value_score": float(frame["value_score"].mean()),
        "mean_roi_priority_score": float(
            frame["roi_priority_score"].mean()
        ),
        "best_value_match": {
            "round_id": int(
                frame.loc[
                    frame["roi_priority_score"].idxmax(),
                    "round_id",
                ]
            ),
            "toto_match_no": int(
                frame.loc[
                    frame["roi_priority_score"].idxmax(),
                    "toto_match_no",
                ]
            ),
            "home_team": str(
                frame.loc[
                    frame["roi_priority_score"].idxmax(),
                    "home_team",
                ]
            ),
            "away_team": str(
                frame.loc[
                    frame["roi_priority_score"].idxmax(),
                    "away_team",
                ]
            ),
            "best_value_pick": str(
                frame.loc[
                    frame["roi_priority_score"].idxmax(),
                    "best_value_pick",
                ]
            ),
            "best_edge": float(
                frame.loc[
                    frame["roi_priority_score"].idxmax(),
                    "best_edge",
                ]
            ),
            "best_value_ratio": float(
                frame.loc[
                    frame["roi_priority_score"].idxmax(),
                    "best_value_ratio",
                ]
            ),
            "roi_priority_score": float(
                frame.loc[
                    frame["roi_priority_score"].idxmax(),
                    "roi_priority_score",
                ]
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(
        description=(
            "Merge current-round v6 AI predictions with normalized "
            "official toto market probabilities."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            root
            / "ml"
            / "current_round"
            / "current_round_predictions_v6.csv"
        ),
    )
    parser.add_argument(
        "--market",
        type=Path,
        default=(
            root
            / "ml"
            / "market_data"
            / "current_toto_market_normalized.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "ml"
            / "current_round"
            / "current_round_market_predictions_v6.csv"
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
            / "market_prediction_merger_v6_summary.json"
        ),
    )
    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=(
            root
            / "ml"
            / "diagnostics"
            / "current_round"
            / "market_value_ranking_v6.csv"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    predictions = prepare_predictions(
        load_csv(args.predictions, "prediction CSV")
    )
    market = prepare_market(
        load_csv(args.market, "normalized market CSV")
    )

    merged = merge_market_and_predictions(
        predictions=predictions,
        market=market,
    )
    summary = build_summary(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.ranking_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    ranking_columns = [
        "round_id",
        "toto_match_no",
        "home_team",
        "away_team",
        "prediction",
        "ai_top_pick",
        "market_top_pick",
        "best_value_pick",
        "best_edge",
        "best_value_ratio",
        "value_score",
        "roi_priority_score",
        "models_agree",
        "ai_market_agree",
    ]
    merged.sort_values(
        "roi_priority_score",
        ascending=False,
    )[ranking_columns].to_csv(
        args.ranking_csv,
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
    print("Project Alpha Market Prediction Merger v6")
    print("=" * 92)
    print(f"Rows                       : {summary['rows']}")
    print(
        f"Duplicate join keys        : "
        f"{summary['duplicate_join_keys']}"
    )
    print(f"Missing values             : {summary['missing_values']}")
    print(
        f"Positive edge rows         : "
        f"{summary['positive_edge_rows']}"
    )
    print(
        f"Strong edge rows           : "
        f"{summary['strong_edge_rows']}"
    )
    print(
        f"Extreme edge rows          : "
        f"{summary['extreme_edge_rows']}"
    )
    print(
        f"AI/Market agreement rows   : "
        f"{summary['ai_market_agreement_rows']}"
    )
    print(
        f"Mean best edge             : "
        f"{summary['mean_best_edge']:.6f}"
    )
    print(
        f"Mean best value ratio      : "
        f"{summary['mean_best_value_ratio']:.6f}"
    )
    print(f"Output                     : {args.output}")
    print(f"Value ranking              : {args.ranking_csv}")
    print(f"Diagnostics                : {args.summary_json}")


if __name__ == "__main__":
    main()
