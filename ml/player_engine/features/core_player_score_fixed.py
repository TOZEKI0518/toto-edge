from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PlayerScore:
    player_key: str
    player_name: str
    team: str
    position: str
    matches_observed: int
    availability: float
    starter_rate: float
    minutes_rate: float
    recent_form: float
    core_score: float


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


def ewma(values: list[float], alpha: float) -> float:
    if not values:
        return 0.0
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return float(result)


def build_scores(
    history: pd.DataFrame,
    window: int = 5,
    ewma_alpha: float = 0.5,
) -> pd.DataFrame:
    required = {"player_name", "team", "position", "starter", "minutes"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = history.copy()
    frame["minutes"] = pd.to_numeric(
        frame["minutes"], errors="coerce"
    ).fillna(0).clip(lower=0, upper=120)
    frame["starter"] = frame["starter"].map(to_bool)

    sort_cols = [
        c for c in ("team", "player_name", "date", "round", "match_card_id")
        if c in frame.columns
    ]
    frame = frame.sort_values(sort_cols)

    rows: list[dict[str, object]] = []

    for (team, player_name), group in frame.groupby(
        ["team", "player_name"], sort=False, dropna=False
    ):
        recent = group.tail(window)
        count = len(recent)
        minutes = recent["minutes"].astype(float).tolist()
        starters = recent["starter"].astype(float).tolist()

        availability = sum(m > 0 for m in minutes) / count
        starter_rate = sum(starters) / count
        minutes_rate = min(sum(minutes) / (90.0 * count), 1.0)
        recent_form = ewma(
            [min(m / 90.0, 1.0) for m in minutes],
            ewma_alpha,
        )

        core_score = (
            0.25 * availability
            + 0.30 * starter_rate
            + 0.25 * minutes_rate
            + 0.20 * recent_form
        )

        position = str(recent["position"].iloc[-1]).strip()
        player_key = f"{str(team).strip()}::{str(player_name).strip()}"

        score = PlayerScore(
            player_key=player_key,
            player_name=str(player_name).strip(),
            team=str(team).strip(),
            position=position,
            matches_observed=count,
            availability=round(availability, 6),
            starter_rate=round(starter_rate, 6),
            minutes_rate=round(minutes_rate, 6),
            recent_form=round(recent_form, 6),
            core_score=round(core_score, 6),
        )
        rows.append(asdict(score))

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["overall_rank"] = (
        result["core_score"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    return result.sort_values(
        ["overall_rank", "team", "player_name"]
    ).reset_index(drop=True)


def main() -> None:
    base = processed_dir()

    parser = argparse.ArgumentParser(
        description="Build lightweight Core Player Scores."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=base / "player_match_history.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "player_core_scores.csv",
    )
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--ewma-alpha", type=float, default=0.5)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    history = pd.read_csv(args.input)
    scores = build_scores(
        history=history,
        window=args.window,
        ewma_alpha=args.ewma_alpha,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("Core Player Score")
    print("=" * 60)
    print(f"Players : {len(scores)}")
    print(f"Saved   : {args.output}")
    if not scores.empty:
        print()
        print(scores.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
