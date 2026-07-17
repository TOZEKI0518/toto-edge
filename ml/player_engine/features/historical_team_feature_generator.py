from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("historical_team_features")

REQUIRED_COLUMNS = {
    "match_card_id",
    "team",
    "player_name",
    "position",
    "starter",
    "minutes",
}

OUTPUT_COLUMNS = [
    "match_card_id",
    "team",
    "match_date",
    "team_core_score",
    "team_momentum",
    "team_stability",
    "core_player_count",
    "available_player_count",
]


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def processed_dir() -> Path:
    """Return the processed-data directory."""
    return Path(__file__).resolve().parents[1] / "data" / "processed"


def to_bool(value: object) -> bool:
    """Normalize common CSV boolean values."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)

    return str(value).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
    }


def ewma(values: list[float], alpha: float) -> float:
    """Calculate an exponentially weighted moving average."""
    if not values:
        return 0.0

    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result

    return float(result)


def validate_columns(frame: pd.DataFrame) -> None:
    """Validate the minimum player history schema."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "player_match_history.csv is missing required columns: "
            f"{sorted(missing)}"
        )


def prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sort player match history."""
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

    if "match_date" not in frame.columns:
        if "date" in frame.columns:
            frame["match_date"] = frame["date"]
        else:
            frame["match_date"] = ""

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
    prior_rows: pd.DataFrame,
    window: int,
    alpha: float,
) -> tuple[float, float]:
    """
    Calculate leakage-free player core and momentum scores.

    Only rows before the target match are supplied.
    """
    if prior_rows.empty:
        return 0.0, 0.0

    recent = prior_rows.tail(window)

    minutes = recent["minutes"].astype(float).tolist()
    starters = recent["starter"].astype(float).tolist()
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

    absence_penalty = min(absence_streak * 0.10, 0.40)

    momentum_score = (
        0.40 * recent_form
        + 0.35 * minutes_rate
        + 0.25 * starter_rate
        - absence_penalty
    )

    return (
        min(max(core_score, 0.0), 1.0),
        min(max(momentum_score, 0.0), 1.0),
    )


def build_historical_team_features(
    history: pd.DataFrame,
    window: int = 5,
    ewma_alpha: float = 0.50,
    core_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Generate match-time team features without future information.

    For each team in each match, player state is calculated exclusively
    from that player's appearances before the target match.
    """
    if window < 1:
        raise ValueError("window must be at least 1.")
    if not 0.0 < ewma_alpha <= 1.0:
        raise ValueError("ewma_alpha must be in (0, 1].")
    if not 0.0 <= core_threshold <= 1.0:
        raise ValueError("core_threshold must be in [0, 1].")

    frame = prepare_history(history)

    player_history: dict[tuple[str, str], list[dict[str, object]]] = {}
    previous_starters: dict[str, set[str]] = {}
    output_rows: list[dict[str, object]] = []

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

        current_rows = frame.loc[
            (frame["match_card_id"] == match_id)
            & (frame["team"] == team)
        ]

        player_scores: list[float] = []
        momentum_scores: list[float] = []
        available_count = 0
        core_count = 0

        for player in current_rows.itertuples(index=False):
            player_name = str(player.player_name)
            key = (team, player_name)

            prior = pd.DataFrame(player_history.get(key, []))

            core_score, momentum_score = calculate_player_state(
                prior_rows=prior,
                window=window,
                alpha=ewma_alpha,
            )

            if not prior.empty:
                available_count += 1

            if core_score >= core_threshold:
                core_count += 1

            player_scores.append(core_score)
            momentum_scores.append(momentum_score)

        current_starters = set(
            current_rows.loc[
                current_rows["starter"],
                "player_name",
            ].astype(str)
        )

        prior_starters = previous_starters.get(team, set())
        if prior_starters:
            team_stability = (
                len(current_starters & prior_starters) / 11.0
            )
        else:
            team_stability = 0.0

        output_rows.append(
            {
                "match_card_id": match_id,
                "team": team,
                "match_date": match_date,
                "team_core_score": round(
                    sum(player_scores) / len(player_scores)
                    if player_scores
                    else 0.0,
                    6,
                ),
                "team_momentum": round(
                    sum(momentum_scores) / len(momentum_scores)
                    if momentum_scores
                    else 0.0,
                    6,
                ),
                "team_stability": round(
                    min(max(team_stability, 0.0), 1.0),
                    6,
                ),
                "core_player_count": core_count,
                "available_player_count": available_count,
            }
        )

        previous_starters[team] = current_starters

        for player in current_rows.itertuples(index=False):
            key = (team, str(player.player_name))
            player_history.setdefault(key, []).append(
                {
                    "minutes": float(player.minutes),
                    "starter": bool(player.starter),
                    "match_card_id": match_id,
                    "match_date": match_date,
                }
            )

    result = pd.DataFrame(output_rows)

    if result.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return result[OUTPUT_COLUMNS].sort_values(
        ["match_date", "match_card_id", "team"],
        kind="stable",
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    base = processed_dir()

    parser = argparse.ArgumentParser(
        description=(
            "Generate leakage-free historical team player features."
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
        default=base / "historical_team_features.csv",
    )
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--ewma-alpha", type=float, default=0.50)
    parser.add_argument("--core-threshold", type=float, default=0.60)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Run the historical feature generation pipeline."""
    args = parse_args()
    configure_logging(args.verbose)

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    history = pd.read_csv(args.input)

    features = build_historical_team_features(
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
