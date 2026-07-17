from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("feature_builder_v4")
POSITIONS = ("GK", "DF", "MF", "FW")

BASE_POSITION_COLUMNS = [
    *(f"core_{position}" for position in POSITIONS),
    *(f"momentum_{position}" for position in POSITIONS),
    *(f"starter_count_{position}" for position in POSITIONS),
    *(f"available_count_{position}" for position in POSITIONS),
]


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_paths() -> tuple[Path, Path, Path]:
    root = project_root()
    processed = root / "ml" / "player_engine" / "data" / "processed"
    return (
        root / "ml" / "training_dataset_v3.csv",
        processed / "historical_position_features.csv",
        root / "ml" / "training_dataset_v4.csv",
    )


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{label} is empty: {path}")
    LOGGER.info("%s rows: %s", label, len(frame))
    return frame


def validate_position_features(frame: pd.DataFrame) -> None:
    required = {"match_card_id", "team", *BASE_POSITION_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "historical_position_features.csv is missing columns: "
            f"{sorted(missing)}"
        )


def prepare_side_features(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    prefix = "home" if side == "home" else "away"
    renamed = frame.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in BASE_POSITION_COLUMNS
        }
    ).copy()
    return renamed[
        [
            "match_card_id",
            "team",
            *[f"{prefix}_{column}" for column in BASE_POSITION_COLUMNS],
        ]
    ].rename(columns={"team": f"{prefix}_team_key"})


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    for position in POSITIONS:
        home_core = f"home_core_{position}"
        away_core = f"away_core_{position}"
        home_momentum = f"home_momentum_{position}"
        away_momentum = f"away_momentum_{position}"

        result[f"{position}CoreDiff"] = result[home_core] - result[away_core]
        result[f"{position}MomentumDiff"] = (
            result[home_momentum] - result[away_momentum]
        )
        result[f"{position}CoreRatio"] = (
            result[home_core] + 0.01
        ) / (
            result[away_core] + 0.01
        )
        result[f"{position}MomentumRatio"] = (
            result[home_momentum] + 0.01
        ) / (
            result[away_momentum] + 0.01
        )

        result[f"home{position}StarterCount"] = result[
            f"home_starter_count_{position}"
        ]
        result[f"away{position}StarterCount"] = result[
            f"away_starter_count_{position}"
        ]
        result[f"{position}StarterCountDiff"] = (
            result[f"home{position}StarterCount"]
            - result[f"away{position}StarterCount"]
        )

        result[f"home{position}AvailableRate"] = (
            result[f"home_available_count_{position}"]
            / result[f"home_starter_count_{position}"].clip(lower=1)
        ).clip(0.0, 1.0)
        result[f"away{position}AvailableRate"] = (
            result[f"away_available_count_{position}"]
            / result[f"away_starter_count_{position}"].clip(lower=1)
        ).clip(0.0, 1.0)
        result[f"{position}AvailableRateDiff"] = (
            result[f"home{position}AvailableRate"]
            - result[f"away{position}AvailableRate"]
        )

    result["homeAttackCore"] = (
        0.70 * result["home_core_FW"]
        + 0.30 * result["home_core_MF"]
    )
    result["awayAttackCore"] = (
        0.70 * result["away_core_FW"]
        + 0.30 * result["away_core_MF"]
    )
    result["attackCoreDiff"] = (
        result["homeAttackCore"] - result["awayAttackCore"]
    )

    result["homeDefenseCore"] = (
        0.75 * result["home_core_DF"]
        + 0.25 * result["home_core_GK"]
    )
    result["awayDefenseCore"] = (
        0.75 * result["away_core_DF"]
        + 0.25 * result["away_core_GK"]
    )
    result["defenseCoreDiff"] = (
        result["homeDefenseCore"] - result["awayDefenseCore"]
    )

    result["homeAttackMomentum"] = (
        0.70 * result["home_momentum_FW"]
        + 0.30 * result["home_momentum_MF"]
    )
    result["awayAttackMomentum"] = (
        0.70 * result["away_momentum_FW"]
        + 0.30 * result["away_momentum_MF"]
    )
    result["attackMomentumDiff"] = (
        result["homeAttackMomentum"] - result["awayAttackMomentum"]
    )

    result["homeDefenseMomentum"] = (
        0.75 * result["home_momentum_DF"]
        + 0.25 * result["home_momentum_GK"]
    )
    result["awayDefenseMomentum"] = (
        0.75 * result["away_momentum_DF"]
        + 0.25 * result["away_momentum_GK"]
    )
    result["defenseMomentumDiff"] = (
        result["homeDefenseMomentum"] - result["awayDefenseMomentum"]
    )

    result["homePlayerBalance"] = (
        result["homeAttackCore"] - result["homeDefenseCore"]
    ).abs()
    result["awayPlayerBalance"] = (
        result["awayAttackCore"] - result["awayDefenseCore"]
    ).abs()
    result["playerBalanceDiff"] = (
        result["homePlayerBalance"] - result["awayPlayerBalance"]
    )

    return result


def build_v4_dataset(
    training_v3: pd.DataFrame,
    position_features: pd.DataFrame,
) -> pd.DataFrame:
    validate_position_features(position_features)

    if "match_card_id" not in training_v3.columns:
        raise ValueError(
            "training_dataset_v3.csv does not contain match_card_id."
        )

    home_team_column = (
        "homeTeam" if "homeTeam" in training_v3.columns else "home_team"
    )
    away_team_column = (
        "awayTeam" if "awayTeam" in training_v3.columns else "away_team"
    )

    if home_team_column not in training_v3.columns:
        raise ValueError("Home team column was not found.")
    if away_team_column not in training_v3.columns:
        raise ValueError("Away team column was not found.")

    position = position_features.copy()
    position["match_card_id"] = pd.to_numeric(
        position["match_card_id"], errors="coerce"
    ).astype("Int64")

    training = training_v3.copy()
    training["match_card_id"] = pd.to_numeric(
        training["match_card_id"], errors="coerce"
    ).astype("Int64")

    home = prepare_side_features(position, "home")
    away = prepare_side_features(position, "away")

    merged = training.merge(
        home,
        left_on=["match_card_id", home_team_column],
        right_on=["match_card_id", "home_team_key"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["home_team_key"])

    merged = merged.merge(
        away,
        left_on=["match_card_id", away_team_column],
        right_on=["match_card_id", "away_team_key"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["away_team_key"])

    position_columns = [
        column
        for column in merged.columns
        if column.startswith("home_") or column.startswith("away_")
    ]

    for column in position_columns:
        merged[column] = pd.to_numeric(
            merged[column], errors="coerce"
        ).fillna(0.0)

    return add_engineered_features(merged)


def parse_args() -> argparse.Namespace:
    training_path, position_path, output_path = default_paths()

    parser = argparse.ArgumentParser(
        description="Build Project Alpha Version 4 training features."
    )
    parser.add_argument("--training", type=Path, default=training_path)
    parser.add_argument(
        "--position-features",
        type=Path,
        default=position_path,
    )
    parser.add_argument("--output", type=Path, default=output_path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    training = load_csv(args.training, "Training dataset v3")
    position = load_csv(
        args.position_features,
        "Historical position features",
    )

    result = build_v4_dataset(training, position)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    added_columns = [
        column for column in result.columns if column not in training.columns
    ]

    matched = int(
        (
            result["home_core_GK"].gt(0)
            | result["away_core_GK"].gt(0)
            | result["home_core_DF"].gt(0)
            | result["away_core_DF"].gt(0)
            | result["home_core_MF"].gt(0)
            | result["away_core_MF"].gt(0)
            | result["home_core_FW"].gt(0)
            | result["away_core_FW"].gt(0)
        ).sum()
    )

    print("=" * 72)
    print("Project Alpha Feature Builder v4")
    print("=" * 72)
    print(f"Rows           : {len(result)}")
    print(f"Matched rows   : {matched}")
    print(f"Columns before : {len(training.columns)}")
    print(f"Columns after  : {len(result.columns)}")
    print(f"Columns added  : {len(added_columns)}")
    print(f"Saved          : {args.output}")
    print()
    print("Added columns")
    print("-" * 72)

    for column in added_columns:
        print(column)


if __name__ == "__main__":
    main()
