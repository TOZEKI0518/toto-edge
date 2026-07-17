from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("historical_position_features")

REQUIRED_COLUMNS = {
    "match_card_id",
    "team",
    "player_name",
    "position",
    "starter",
    "minutes",
}

POSITIONS = ("GK", "DF", "MF", "FW")

OUTPUT_COLUMNS = [
    "match_card_id",
    "team",
    "match_date",
    *[
        f"{prefix}{position}"
        for position in POSITIONS
        for prefix in (
            "core_",
            "momentum_",
            "starter_count_",
            "available_count_",
        )
    ],
]


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def processed_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "processed"


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def normalize_position(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return text if text in POSITIONS else ""


def ewma(values: list[float], alpha: float) -> float:
    if not values:
        return 0.0

    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return float(result)


def validate_columns(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "player_match_history.csv is missing required columns: "
            f"{sorted(missing)}"
        )


def prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    validate_columns(history)

    frame = history.copy()

    frame["match_card_id"] = pd.to_numeric(
        frame["match_card_id"],
        errors="raise",
    ).astype(int)

    frame["minutes"] = pd.to_numeric(
        frame["minutes"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0, upper=120.0)

    frame["starter"] = frame["starter"].map(to_bool)
    frame["position"] = frame["position"].map(normalize_position)

    if "match_date" not in frame.columns:
        frame["match_date"] = (
            frame["date"]
            if "date" in frame.columns
            else ""
        )

    frame["match_date"] = frame["match_date"].fillna("").astype(str)

    sort_columns = [
        column
        for column in (
            "match_date",
            "round",
            "match_card_id",
            "team",
            "player_name",
        )
        if column in frame.columns
    ]

    return frame.sort_values(sort_columns).reset_index(drop=True)


def calculate_player_state(
    prior_rows: list[dict[str, object]],
    window: int,
    alpha: float,
) -> tuple[float, float]:
    if not prior_rows:
        return 0.0, 0.0

    recent = prior_rows[-window:]

    minutes = [float(row["minutes"]) for row in recent]
    starters = [1.0 if bool(row["starter"]) else 0.0 for row in recent]
    count = len(recent)

    availability = sum(value > 0 for value in minutes) / count
    starter_rate = sum(starters) / count
    minutes_rate = min(sum(minutes) / (90.0 * count), 1.0)

    normalized_minutes = [
        min(max(value / 90.0, 0.0), 1.0)
        for value in minutes
    ]
    recent_form = ewma(normalized_minutes, alpha)

    core_score = (
        0.25 * availability
        + 0.30 * starter_rate
        + 0.25 * minutes_rate
        + 0.20 * recent_form
    )

    absence_streak = 0
    for value in reversed(minutes):
        if value > 0:
            break
        absence_streak += 1

    momentum_score = (
        0.40 * recent_form
        + 0.35 * minutes_rate
        + 0.25 * starter_rate
        - min(absence_streak * 0.10, 0.40)
    )

    return (
        min(max(core_score, 0.0), 1.0),
        min(max(momentum_score, 0.0), 1.0),
    )


def build_historical_position_features(
    history: pd.DataFrame,
    window: int = 5,
    ewma_alpha: float = 0.50,
    core_threshold: float = 0.60,
) -> pd.DataFrame:
    if window < 1:
        raise ValueError("window must be at least 1.")
    if not 0.0 < ewma_alpha <= 1.0:
        raise ValueError("ewma_alpha must be in (0, 1].")
    if not 0.0 <= core_threshold <= 1.0:
        raise ValueError("core_threshold must be in [0, 1].")

    frame = prepare_history(history)

    player_history: dict[tuple[str, str], list[dict[str, object]]] = {}
    rows: list[dict[str, object]] = []

    match_keys = (
        frame[["match_card_id", "team", "match_date"]]
        .drop_duplicates()
        .sort_values(
            ["match_date", "match_card_id", "team"],
            kind="stable",
        )
    )

    for match in match_keys.itertuples(index=False):
        match_id = int(match.match_card_id)
        team = str(match.team)
        match_date = str(match.match_date)

        current = frame.loc[
            (frame["match_card_id"] == match_id)
            & (frame["team"] == team)
        ]

        result: dict[str, object] = {
            "match_card_id": match_id,
            "team": team,
            "match_date": match_date,
        }

        aggregates: dict[str, dict[str, float]] = {
            position: {
                "core_sum": 0.0,
                "momentum_sum": 0.0,
                "player_count": 0.0,
                "starter_count": 0.0,
                "available_count": 0.0,
            }
            for position in POSITIONS
        }

        for player in current.itertuples(index=False):
            position = normalize_position(player.position)
            if position not in POSITIONS:
                continue

            key = (team, str(player.player_name))
            prior_rows = player_history.get(key, [])

            core_score, momentum_score = calculate_player_state(
                prior_rows=prior_rows,
                window=window,
                alpha=ewma_alpha,
            )

            aggregates[position]["core_sum"] += core_score
            aggregates[position]["momentum_sum"] += momentum_score
            aggregates[position]["player_count"] += 1

            if bool(player.starter):
                aggregates[position]["starter_count"] += 1

            if prior_rows and core_score >= core_threshold:
                aggregates[position]["available_count"] += 1

        for position in POSITIONS:
            item = aggregates[position]
            count = item["player_count"]

            result[f"core_{position}"] = round(
                item["core_sum"] / count if count else 0.0,
                6,
            )
            result[f"momentum_{position}"] = round(
                item["momentum_sum"] / count if count else 0.0,
                6,
            )
            result[f"starter_count_{position}"] = int(
                item["starter_count"]
            )
            result[f"available_count_{position}"] = int(
                item["available_count"]
            )

        rows.append(result)

        for player in current.itertuples(index=False):
            key = (team, str(player.player_name))
            player_history.setdefault(key, []).append(
                {
                    "minutes": float(player.minutes),
                    "starter": bool(player.starter),
                    "position": normalize_position(player.position),
                    "match_card_id": match_id,
                    "match_date": match_date,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return result[OUTPUT_COLUMNS].sort_values(
        ["match_date", "match_card_id", "team"],
        kind="stable",
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    base = processed_dir()

    parser = argparse.ArgumentParser(
        description=(
            "Generate leakage-free historical position player features."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=base / "player_match_history.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "historical_position_features.csv",
    )
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--ewma-alpha", type=float, default=0.50)
    parser.add_argument("--core-threshold", type=float, default=0.60)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    history = pd.read_csv(args.input)

    features = build_historical_position_features(
        history=history,
        window=args.window,
        ewma_alpha=args.ewma_alpha,
        core_threshold=args.core_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    LOGGER.info("Team-match rows: %s", len(features))
    LOGGER.info(
        "Matches: %s",
        features["match_card_id"].nunique()
        if not features.empty
        else 0,
    )
    LOGGER.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
