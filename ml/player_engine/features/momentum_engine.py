from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("momentum_engine")

REQUIRED_COLUMNS = {
    "player_name",
    "team",
    "starter",
    "minutes",
}

OUTPUT_COLUMNS = [
    "player_key",
    "player_name",
    "team",
    "position",
    "matches_observed",
    "minutes_last5",
    "starter_rate_last5",
    "minutes_ewma",
    "absence_streak",
    "momentum_score",
]


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def default_processed_dir() -> Path:
    """Return the Project Alpha processed-data directory."""
    return Path(__file__).resolve().parents[1] / "data" / "processed"


def validate_columns(frame: pd.DataFrame) -> None:
    """Validate the minimum player history schema."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "player_match_history.csv is missing required columns: "
            f"{sorted(missing)}"
        )


def normalize_boolean(value: object) -> bool:
    """Convert common CSV representations to bool."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False

    raise ValueError(f"Invalid starter value: {value!r}")


def ewma(values: list[float], alpha: float) -> float:
    """Calculate an exponentially weighted moving average."""
    if not values:
        return 0.0

    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return float(result)


def calculate_absence_streak(minutes: list[float]) -> int:
    """Count consecutive zero-minute appearances from the latest match."""
    streak = 0
    for value in reversed(minutes):
        if value > 0:
            break
        streak += 1
    return streak


def build_player_momentum(
    history: pd.DataFrame,
    window: int = 5,
    ewma_alpha: float = 0.50,
) -> pd.DataFrame:
    """
    Build lightweight player momentum features.

    momentum_score:
        40% recent minutes EWMA
        35% recent minutes share
        25% recent starter rate
        minus an absence-streak penalty
    """
    if window < 1:
        raise ValueError("window must be at least 1.")
    if not 0.0 < ewma_alpha <= 1.0:
        raise ValueError("ewma_alpha must be in (0, 1].")

    validate_columns(history)

    frame = history.copy()
    frame["minutes"] = pd.to_numeric(
        frame["minutes"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0, upper=120.0)

    frame["starter"] = frame["starter"].map(normalize_boolean)

    sort_columns = [
        column
        for column in ("player_name", "date", "round", "match_card_id")
        if column in frame.columns
    ]
    if "player_name" not in sort_columns:
        sort_columns.insert(0, "player_name")

    frame = frame.sort_values(sort_columns)

    rows: list[dict[str, object]] = []

    group_columns = ["team", "player_name"]
    for (team, player_name), player_frame in frame.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        recent = player_frame.tail(window)

        minutes = recent["minutes"].astype(float).tolist()
        starters = recent["starter"].astype(float).tolist()

        minutes_last5 = sum(minutes)
        starter_rate_last5 = (
            sum(starters) / len(starters) if starters else 0.0
        )
        minutes_share = (
            minutes_last5 / (90.0 * len(minutes)) if minutes else 0.0
        )
        minutes_share = min(max(minutes_share, 0.0), 1.0)

        normalized_minutes = [
            min(max(value / 90.0, 0.0), 1.0)
            for value in minutes
        ]
        minutes_ewma = ewma(normalized_minutes, ewma_alpha)
        absence_streak = calculate_absence_streak(minutes)

        absence_penalty = min(absence_streak * 0.10, 0.40)

        momentum_score = (
            0.40 * minutes_ewma
            + 0.35 * minutes_share
            + 0.25 * starter_rate_last5
            - absence_penalty
        )
        momentum_score = min(max(momentum_score, 0.0), 1.0)

        position = ""
        if "position" in recent.columns:
            position_values = recent["position"].dropna().astype(str)
            if not position_values.empty:
                position = position_values.iloc[-1]

        player_key = f"{str(team).strip()}::{str(player_name).strip()}"

        rows.append(
            {
                "player_key": player_key,
                "player_name": str(player_name).strip(),
                "team": str(team).strip(),
                "position": position,
                "matches_observed": len(recent),
                "minutes_last5": round(minutes_last5, 2),
                "starter_rate_last5": round(starter_rate_last5, 6),
                "minutes_ewma": round(minutes_ewma, 6),
                "absence_streak": absence_streak,
                "momentum_score": round(momentum_score, 6),
            }
        )

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return pd.DataFrame(rows)[OUTPUT_COLUMNS].sort_values(
        ["team", "momentum_score", "player_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    processed_dir = default_processed_dir()

    parser = argparse.ArgumentParser(
        description="Build lightweight player momentum features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=processed_dir / "player_match_history.csv",
        help="Input player match history CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=processed_dir / "player_momentum.csv",
        help="Output player momentum CSV.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Number of recent matches per player.",
    )
    parser.add_argument(
        "--ewma-alpha",
        type=float,
        default=0.50,
        help="EWMA alpha for recent playing time.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the player momentum pipeline."""
    args = parse_args()
    configure_logging(args.verbose)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    history = pd.read_csv(args.input)

    features = build_player_momentum(
        history=history,
        window=args.window,
        ewma_alpha=args.ewma_alpha,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    LOGGER.info("Players: %s", len(features))
    LOGGER.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
