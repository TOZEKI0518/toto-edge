from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    ML_ROOT = Path(__file__).resolve().parent
    if str(ML_ROOT) not in sys.path:
        sys.path.insert(0, str(ML_ROOT))

from head_to_head.feature_builder import HeadToHeadFeatureBuilder
from head_to_head.models import (
    HeadToHeadEngineConfig,
    MatchOutcome,
    MatchRecord,
)
from head_to_head.repository import HeadToHeadRepository, RepositoryConfig

LOGGER = logging.getLogger("current_round_feature_generator_v6")

TEAM_FEATURES = (
    "team_core_score",
    "team_momentum",
    "team_stability",
    "core_player_count",
    "available_player_count",
)

POSITIONS = ("GK", "DF", "MF", "FW")
POSITION_METRICS = (
    "core",
    "momentum",
    "starter_count",
    "available_count",
)

PAIR_FEATURE_BASES = (
    *TEAM_FEATURES,
    *(
        f"{metric}_{position}"
        for position in POSITIONS
        for metric in POSITION_METRICS
    ),
)

IDENTIFIER_COLUMNS = (
    "match_card_id",
    "match_date",
    "home_team",
    "away_team",
    "season",
    "competition",
    "round",
)

MARKET_COLUMNS = (
    "round_id",
    "toto_match_no",
    "market_prob_home",
    "market_prob_draw",
    "market_prob_away",
)


class CurrentRoundFeatureError(RuntimeError):
    """Raised when current-round features cannot be generated safely."""


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def default_paths() -> dict[str, Path]:
    ml_root = Path(__file__).resolve().parent
    processed = ml_root / "player_engine" / "data" / "processed"
    return {
        "market_csv": (
            ml_root
            / "market_data"
            / "current_toto_market_normalized.csv"
        ),
        "matches_csv": processed / "matches.csv",
        "team_features_csv": processed / "historical_team_features.csv",
        "position_features_csv": (
            processed / "historical_position_features.csv"
        ),
        "training_schema_csv": ml_root / "training_dataset_v3.csv",
        "output_csv": (
            ml_root
            / "current_round"
            / "current_round_features_v6.csv"
        ),
        "diagnostics_json": (
            ml_root
            / "diagnostics"
            / "current_round"
            / "current_round_features_v6_summary.json"
        ),
        "missing_csv": (
            ml_root
            / "diagnostics"
            / "current_round"
            / "current_round_missing_features.csv"
        ),
    }


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)
    frame = pd.read_csv(path)
    if frame.empty:
        raise CurrentRoundFeatureError(f"{label} is empty: {path}")
    LOGGER.info(
        "Loaded %s: rows=%d columns=%d",
        label,
        len(frame),
        len(frame.columns),
    )
    return frame


def restore_feature_dates(
    feature_frame: pd.DataFrame,
    matches_frame: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Restore missing feature dates from matches.csv via match_card_id."""
    required_feature_columns = {"match_card_id", "match_date"}
    missing_feature_columns = sorted(
        required_feature_columns - set(feature_frame.columns)
    )
    if missing_feature_columns:
        raise CurrentRoundFeatureError(
            f"{label} is missing columns required for date restoration: "
            + ", ".join(missing_feature_columns)
        )

    required_match_columns = {"match_card_id", "match_date"}
    missing_match_columns = sorted(
        required_match_columns - set(matches_frame.columns)
    )
    if missing_match_columns:
        raise CurrentRoundFeatureError(
            "processed matches CSV is missing columns: "
            + ", ".join(missing_match_columns)
        )

    restored = feature_frame.copy()
    restored["match_date"] = pd.to_datetime(
        restored["match_date"],
        errors="coerce",
    )

    date_lookup = matches_frame[
        ["match_card_id", "match_date"]
    ].copy()
    date_lookup["match_date"] = pd.to_datetime(
        date_lookup["match_date"],
        errors="coerce",
    )
    date_lookup = date_lookup.dropna(
        subset=["match_card_id", "match_date"]
    ).drop_duplicates(
        subset=["match_card_id"],
        keep="last",
    )
    lookup = date_lookup.set_index("match_card_id")["match_date"]

    missing_before = int(restored["match_date"].isna().sum())
    restored.loc[
        restored["match_date"].isna(),
        "match_date",
    ] = restored.loc[
        restored["match_date"].isna(),
        "match_card_id",
    ].map(lookup)

    missing_after = int(restored["match_date"].isna().sum())
    LOGGER.info(
        "%s date restoration: missing_before=%d missing_after=%d",
        label,
        missing_before,
        missing_after,
    )

    if missing_after:
        raise CurrentRoundFeatureError(
            f"{label} still contains {missing_after} missing match dates "
            "after restoration from matches.csv."
        )

    return restored


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def resolve_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
    label: str = "column",
) -> str | None:
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        matched = by_lower.get(candidate.lower())
        if matched is not None:
            return matched

    if required:
        raise CurrentRoundFeatureError(
            f"Could not resolve {label}. Candidates: {list(candidates)}"
        )
    return None


def infer_match_year(match_day: str, collected_at: str) -> int:
    """Infer year by selecting the date closest to collection timestamp."""
    collected = pd.to_datetime(collected_at, errors="coerce")
    if pd.isna(collected):
        collected = pd.Timestamp.now()

    day_match = pd.to_datetime(
        match_day,
        format="%m/%d",
        errors="coerce",
    )
    if pd.isna(day_match):
        return int(collected.year)

    month = int(day_match.month)
    day = int(day_match.day)
    candidate_years = (
        int(collected.year) - 1,
        int(collected.year),
        int(collected.year) + 1,
    )
    candidates = [
        pd.Timestamp(year=year, month=month, day=day)
        for year in candidate_years
    ]
    closest = min(
        candidates,
        key=lambda value: abs((value - collected).days),
    )
    return int(closest.year)


def prepare_market_frame(market: pd.DataFrame) -> pd.DataFrame:
    required = {
        "round_id",
        "toto_match_no",
        "home_team",
        "away_team",
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
    }
    missing = sorted(required - set(market.columns))
    if missing:
        raise CurrentRoundFeatureError(
            "Normalized market CSV is missing columns: "
            + ", ".join(missing)
        )

    if len(market) != 13:
        raise CurrentRoundFeatureError(
            f"Expected 13 market rows, found {len(market)}."
        )

    market = market.sort_values(
        "toto_match_no",
        kind="stable",
    ).reset_index(drop=True)

    expected_numbers = list(range(1, 14))
    actual_numbers = market["toto_match_no"].astype(int).tolist()
    if actual_numbers != expected_numbers:
        raise CurrentRoundFeatureError(
            f"Expected toto_match_no 1..13, found {actual_numbers}."
        )

    if market[["home_team", "away_team"]].isna().any().any():
        raise CurrentRoundFeatureError("Market team names contain missing values.")

    if "match_day" not in market.columns:
        market["match_day"] = ""
    if "collected_at" not in market.columns:
        market["collected_at"] = datetime.now().isoformat()

    dates: list[str] = []
    for row in market.itertuples(index=False):
        match_day = normalize_text(getattr(row, "match_day", ""))
        collected_at = normalize_text(
            getattr(row, "collected_at", "")
        )
        year = infer_match_year(match_day, collected_at)
        parsed = pd.to_datetime(
            f"{year}/{match_day}",
            format="%Y/%m/%d",
            errors="coerce",
        )
        if pd.isna(parsed):
            parsed = pd.to_datetime(collected_at, errors="coerce")
        if pd.isna(parsed):
            raise CurrentRoundFeatureError(
                f"Could not determine match date for toto match "
                f"{getattr(row, 'toto_match_no')}."
            )
        dates.append(parsed.strftime("%Y-%m-%d"))

    market["match_date"] = dates
    market["season"] = pd.to_datetime(market["match_date"]).dt.year
    market["competition"] = "J.League"
    market["round"] = market["round_id"]

    round_ids = market["round_id"].astype(int)
    match_numbers = market["toto_match_no"].astype(int)
    market["match_card_id"] = round_ids * 100 + match_numbers

    probability_columns = [
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
    ]
    probabilities = market[probability_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if probabilities.isna().any().any():
        raise CurrentRoundFeatureError(
            "Market probabilities contain missing/non-numeric values."
        )
    totals = probabilities.sum(axis=1)
    if (totals <= 0).any():
        raise CurrentRoundFeatureError(
            "Market probability row sum must be positive."
        )
    market[probability_columns] = probabilities.div(totals, axis=0)

    return market


def prepare_snapshot_frame(
    frame: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, str, str]:
    team_column = resolve_column(
        frame,
        (
            "team_name",
            "team",
            "club_name",
            "club",
        ),
        label=f"{label} team column",
    )
    date_column = resolve_column(
        frame,
        (
            "match_date",
            "snapshot_date",
            "feature_date",
            "date",
        ),
        label=f"{label} date column",
    )

    prepared = frame.copy()
    prepared[team_column] = prepared[team_column].astype(str).str.strip()
    prepared[date_column] = pd.to_datetime(
        prepared[date_column],
        errors="coerce",
    )
    prepared = prepared.loc[
        prepared[team_column].ne("")
        & prepared[date_column].notna()
    ].copy()
    prepared = prepared.sort_values(
        [team_column, date_column],
        kind="stable",
    )
    return prepared, team_column, date_column


def latest_team_snapshot(
    frame: pd.DataFrame,
    team_column: str,
    date_column: str,
    team_name: str,
    before_date: pd.Timestamp,
) -> pd.Series | None:
    candidates = frame.loc[
        frame[team_column].eq(team_name)
        & frame[date_column].lt(before_date)
    ]
    if candidates.empty:
        return None
    return candidates.iloc[-1]


def numeric_from_candidates(
    row: pd.Series | None,
    candidates: Iterable[str],
    *,
    default: float = np.nan,
) -> float:
    if row is None:
        return float(default)

    row_index_lower = {
        str(column).lower(): column
        for column in row.index
    }
    for candidate in candidates:
        column = (
            candidate
            if candidate in row.index
            else row_index_lower.get(candidate.lower())
        )
        if column is None:
            continue
        value = pd.to_numeric(
            pd.Series([row[column]]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(value):
            return float(value)

    return float(default)


def team_metric_candidates(metric: str) -> tuple[str, ...]:
    aliases = {
        "team_core_score": (
            "team_core_score",
            "core_score",
            "team_score",
        ),
        "team_momentum": (
            "team_momentum",
            "momentum",
            "momentum_score",
        ),
        "team_stability": (
            "team_stability",
            "stability",
            "stability_score",
        ),
        "core_player_count": (
            "core_player_count",
            "core_count",
        ),
        "available_player_count": (
            "available_player_count",
            "available_count",
            "squad_available_count",
        ),
    }
    return aliases[metric]


def position_metric_candidates(
    metric: str,
    position: str,
) -> tuple[str, ...]:
    return (
        f"{metric}_{position}",
        f"{position}_{metric}",
        f"{metric.lower()}_{position.lower()}",
        f"{position.lower()}_{metric.lower()}",
    )


def add_side_snapshot_features(
    output: dict[str, object],
    *,
    side: str,
    team_row: pd.Series | None,
    position_row: pd.Series | None,
) -> None:
    for metric in TEAM_FEATURES:
        output[f"{side}_{metric}"] = numeric_from_candidates(
            team_row,
            team_metric_candidates(metric),
        )

    for position in POSITIONS:
        for metric in POSITION_METRICS:
            output[f"{side}_{metric}_{position}"] = (
                numeric_from_candidates(
                    position_row,
                    position_metric_candidates(metric, position),
                )
            )


def safe_ratio(home: float, away: float) -> float:
    if pd.isna(home) or pd.isna(away):
        return np.nan
    if abs(away) < 1e-12:
        return np.nan
    return float(home / away)


def add_pair_features(output: dict[str, object]) -> None:
    for base in PAIR_FEATURE_BASES:
        home_column = f"home_{base}"
        away_column = f"away_{base}"
        home_value = pd.to_numeric(
            pd.Series([output.get(home_column)]),
            errors="coerce",
        ).iloc[0]
        away_value = pd.to_numeric(
            pd.Series([output.get(away_column)]),
            errors="coerce",
        ).iloc[0]

        output[f"{base}Diff"] = (
            float(home_value - away_value)
            if pd.notna(home_value) and pd.notna(away_value)
            else np.nan
        )
        output[f"{base}Ratio"] = safe_ratio(
            float(home_value) if pd.notna(home_value) else np.nan,
            float(away_value) if pd.notna(away_value) else np.nan,
        )


def make_h2h_builder(matches_csv: Path) -> HeadToHeadFeatureBuilder:
    repository = HeadToHeadRepository(
        RepositoryConfig(matches_csv=matches_csv)
    )
    repository.load()

    dummy_output = matches_csv.parent / "_unused_h2h_output.csv"
    diagnostics = matches_csv.parent / "_unused_h2h_diagnostics"
    config = HeadToHeadEngineConfig(
        input_matches_csv=matches_csv,
        output_features_csv=dummy_output,
        diagnostics_dir=diagnostics,
    )
    return HeadToHeadFeatureBuilder(repository, config)


def current_match_record(row: pd.Series) -> MatchRecord:
    return MatchRecord(
        match_card_id=int(row["match_card_id"]),
        match_date=pd.Timestamp(row["match_date"]).date(),
        season=int(row["season"]),
        competition=str(row["competition"]),
        home_team=str(row["home_team"]),
        away_team=str(row["away_team"]),
        home_score=0.0,
        away_score=0.0,
        outcome=MatchOutcome.DRAW,
    )


def build_current_features(
    market: pd.DataFrame,
    team_features: pd.DataFrame,
    position_features: pd.DataFrame,
    matches_csv: Path,
) -> pd.DataFrame:
    team_prepared, team_name_col, team_date_col = (
        prepare_snapshot_frame(team_features, "team features")
    )
    position_prepared, position_name_col, position_date_col = (
        prepare_snapshot_frame(position_features, "position features")
    )
    h2h_builder = make_h2h_builder(matches_csv)

    rows: list[dict[str, object]] = []

    for market_row in market.to_dict(orient="records"):
        target_date = pd.Timestamp(market_row["match_date"])
        home_team = str(market_row["home_team"])
        away_team = str(market_row["away_team"])

        home_team_snapshot = latest_team_snapshot(
            team_prepared,
            team_name_col,
            team_date_col,
            home_team,
            target_date,
        )
        away_team_snapshot = latest_team_snapshot(
            team_prepared,
            team_name_col,
            team_date_col,
            away_team,
            target_date,
        )
        home_position_snapshot = latest_team_snapshot(
            position_prepared,
            position_name_col,
            position_date_col,
            home_team,
            target_date,
        )
        away_position_snapshot = latest_team_snapshot(
            position_prepared,
            position_name_col,
            position_date_col,
            away_team,
            target_date,
        )

        output: dict[str, object] = {
            column: market_row.get(column)
            for column in (
                *IDENTIFIER_COLUMNS,
                *MARKET_COLUMNS,
            )
        }
        output["team_feature_matched"] = int(
            home_team_snapshot is not None
            and away_team_snapshot is not None
        )
        output["position_feature_matched"] = int(
            home_position_snapshot is not None
            and away_position_snapshot is not None
        )

        add_side_snapshot_features(
            output,
            side="home",
            team_row=home_team_snapshot,
            position_row=home_position_snapshot,
        )
        add_side_snapshot_features(
            output,
            side="away",
            team_row=away_team_snapshot,
            position_row=away_position_snapshot,
        )
        add_pair_features(output)

        match_record = current_match_record(pd.Series(market_row))
        h2h_row = h2h_builder.build(match_record)
        output.update(asdict(h2h_row))

        rows.append(output)

    return pd.DataFrame(rows)


def align_to_training_schema(
    current: pd.DataFrame,
    training_schema: pd.DataFrame,
) -> pd.DataFrame:
    """Align AI columns to v3 while retaining current market metadata."""
    excluded_training_columns = {
        "home_score",
        "away_score",
        "result",
    }
    training_columns = [
        column
        for column in training_schema.columns
        if column not in excluded_training_columns
    ]

    market_extra_columns = [
        column
        for column in (
            "round_id",
            "toto_match_no",
            "market_prob_home",
            "market_prob_draw",
            "market_prob_away",
        )
        if column in current.columns
    ]

    for column in training_columns:
        if column not in current.columns:
            current[column] = np.nan

    ordered = [
        *training_columns,
        *(
            column
            for column in market_extra_columns
            if column not in training_columns
        ),
    ]
    return current[ordered].copy()


def summarize(
    output: pd.DataFrame,
    training_schema: pd.DataFrame,
) -> dict[str, object]:
    model_feature_columns = [
        column
        for column in training_schema.columns
        if column not in {
            "match_card_id",
            "match_date",
            "home_team",
            "away_team",
            "season",
            "competition",
            "round",
            "home_score",
            "away_score",
            "result",
        }
        and pd.api.types.is_numeric_dtype(training_schema[column])
    ]

    missing_by_column = (
        output[model_feature_columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    return {
        "rows": int(len(output)),
        "columns": int(len(output.columns)),
        "duplicate_match_card_ids": int(
            output["match_card_id"].duplicated().sum()
        ),
        "team_feature_matched_rows": int(
            output["team_feature_matched"].sum()
        ),
        "position_feature_matched_rows": int(
            output["position_feature_matched"].sum()
        ),
        "model_feature_columns": int(len(model_feature_columns)),
        "model_feature_missing_values": int(
            output[model_feature_columns].isna().sum().sum()
        ),
        "fully_complete_model_rows": int(
            output[model_feature_columns].notna().all(axis=1).sum()
        ),
        "top_missing_columns": {
            str(column): int(value)
            for column, value in missing_by_column.head(20).items()
            if int(value) > 0
        },
        "date_min": str(output["match_date"].min()),
        "date_max": str(output["match_date"].max()),
        "round_ids": sorted(
            output["round_id"].astype(int).unique().tolist()
        ),
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Generate Project Alpha v6 AI features for the current "
            "13-match toto round."
        )
    )
    parser.add_argument(
        "--market-csv",
        type=Path,
        default=paths["market_csv"],
    )
    parser.add_argument(
        "--matches-csv",
        type=Path,
        default=paths["matches_csv"],
    )
    parser.add_argument(
        "--team-features-csv",
        type=Path,
        default=paths["team_features_csv"],
    )
    parser.add_argument(
        "--position-features-csv",
        type=Path,
        default=paths["position_features_csv"],
    )
    parser.add_argument(
        "--training-schema-csv",
        type=Path,
        default=paths["training_schema_csv"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths["output_csv"],
    )
    parser.add_argument(
        "--diagnostics-json",
        type=Path,
        default=paths["diagnostics_json"],
    )
    parser.add_argument(
        "--missing-csv",
        type=Path,
        default=paths["missing_csv"],
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    market = prepare_market_frame(
        read_csv(args.market_csv, "normalized market CSV")
    )
    team_features = read_csv(
        args.team_features_csv,
        "historical team features",
    )
    position_features = read_csv(
        args.position_features_csv,
        "historical position features",
    )
    training_schema = read_csv(
        args.training_schema_csv,
        "training dataset v3 schema",
    )
    matches_frame = read_csv(
        args.matches_csv,
        "processed matches CSV",
    )

    team_features = restore_feature_dates(
        feature_frame=team_features,
        matches_frame=matches_frame,
        label="historical team features",
    )
    position_features = restore_feature_dates(
        feature_frame=position_features,
        matches_frame=matches_frame,
        label="historical position features",
    )

    current = build_current_features(
        market=market,
        team_features=team_features,
        position_features=position_features,
        matches_csv=args.matches_csv,
    )
    output = align_to_training_schema(
        current=current,
        training_schema=training_schema,
    )

    if len(output) != 13:
        raise CurrentRoundFeatureError(
            f"Expected 13 output rows, found {len(output)}."
        )
    if output["match_card_id"].duplicated().any():
        raise CurrentRoundFeatureError(
            "Output contains duplicate match_card_id values."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.missing_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize(output, training_schema)
    args.diagnostics_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "column": output.columns,
            "missing_count": [
                int(output[column].isna().sum())
                for column in output.columns
            ],
            "dtype": [
                str(output[column].dtype)
                for column in output.columns
            ],
        }
    ).sort_values(
        ["missing_count", "column"],
        ascending=[False, True],
    ).to_csv(
        args.missing_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 88)
    print("Project Alpha Current Round Feature Generator v6")
    print("=" * 88)
    print(f"Rows                         : {summary['rows']}")
    print(f"Columns                      : {summary['columns']}")
    print(
        f"Duplicate match IDs          : "
        f"{summary['duplicate_match_card_ids']}"
    )
    print(
        f"Team feature matched rows    : "
        f"{summary['team_feature_matched_rows']}"
    )
    print(
        f"Position feature matched rows: "
        f"{summary['position_feature_matched_rows']}"
    )
    print(
        f"Model feature columns        : "
        f"{summary['model_feature_columns']}"
    )
    print(
        f"Model feature missing values : "
        f"{summary['model_feature_missing_values']}"
    )
    print(
        f"Fully complete model rows    : "
        f"{summary['fully_complete_model_rows']}"
    )
    print(f"Output                       : {args.output}")
    print(f"Diagnostics                  : {args.diagnostics_json}")


if __name__ == "__main__":
    main()
