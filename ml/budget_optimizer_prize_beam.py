from __future__ import annotations

from pathlib import Path
import argparse
from dataclasses import dataclass

import pandas as pd


INPUT_PATH = Path("ml/round_predictions.csv")
OUTPUT_PATH = Path("ml/budget_ticket_plan_prize_beam.csv")

UNIT_PRICE = 100


@dataclass
class BeamState:
    sizes: list[int]
    coverages: list[float]
    combinations: int
    distribution: list[float]
    score: float


def ranked_outcomes(
    row: pd.Series,
) -> list[tuple[str, float]]:
    outcomes = [
        ("H", float(row["probH"])),
        ("D", float(row["probD"])),
        ("A", float(row["probA"])),
    ]

    return sorted(
        outcomes,
        key=lambda item: item[1],
        reverse=True,
    )


def ticket_from_size(
    row: pd.Series,
    size: int,
) -> tuple[str, str, float]:
    ranked = ranked_outcomes(row)

    if size == 1:
        return (
            ranked[0][0],
            "Single",
            ranked[0][1],
        )

    if size == 2:
        return (
            ranked[0][0] + ranked[1][0],
            "Double",
            ranked[0][1] + ranked[1][1],
        )

    if size == 3:
        return (
            "HDA",
            "Triple",
            1.0,
        )

    raise ValueError(
        f"Unsupported ticket size: {size}"
    )


def update_distribution(
    distribution: list[float],
    coverage: float,
) -> list[float]:
    """
    distribution[k] =
    現時点までにk試合をカバーできる確率
    """
    next_distribution = [
        0.0
    ] * (len(distribution) + 1)

    for hits, probability in enumerate(
        distribution
    ):
        next_distribution[hits] += (
            probability
            * (1.0 - coverage)
        )

        next_distribution[hits + 1] += (
            probability
            * coverage
        )

    return next_distribution


def partial_prize_score(
    distribution: list[float],
    matches_processed: int,
    exact_weight: float,
    one_miss_weight: float,
    two_miss_weight: float,
) -> float:
    """
    途中段階でも比較できるように、

    全的中
    1試合外れ
    2試合外れ

    の確率を加重評価する。

    13試合終了時には、
    13/13・12/13・11/13に対応する。
    """
    all_correct = distribution[
        matches_processed
    ]

    one_miss = (
        distribution[
            matches_processed - 1
        ]
        if matches_processed >= 1
        else 0.0
    )

    two_miss = (
        distribution[
            matches_processed - 2
        ]
        if matches_processed >= 2
        else 0.0
    )

    return (
        exact_weight * all_correct
        + one_miss_weight * one_miss
        + two_miss_weight * two_miss
    )


def beam_search(
    df: pd.DataFrame,
    max_tickets: int,
    beam_width: int,
    exact_weight: float,
    one_miss_weight: float,
    two_miss_weight: float,
) -> BeamState:
    states = [
        BeamState(
            sizes=[],
            coverages=[],
            combinations=1,
            distribution=[1.0],
            score=1.0,
        )
    ]

    for match_index, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        next_states: list[BeamState] = []

        for state in states:
            for size in (1, 2, 3):
                new_combinations = (
                    state.combinations
                    * size
                )

                if (
                    new_combinations
                    > max_tickets
                ):
                    continue

                (
                    _,
                    _,
                    coverage,
                ) = ticket_from_size(
                    row,
                    size,
                )

                new_distribution = (
                    update_distribution(
                        distribution=(
                            state.distribution
                        ),
                        coverage=coverage,
                    )
                )

                new_score = (
                    partial_prize_score(
                        distribution=(
                            new_distribution
                        ),
                        matches_processed=(
                            match_index
                        ),
                        exact_weight=(
                            exact_weight
                        ),
                        one_miss_weight=(
                            one_miss_weight
                        ),
                        two_miss_weight=(
                            two_miss_weight
                        ),
                    )
                )

                next_states.append(
                    BeamState(
                        sizes=(
                            state.sizes
                            + [size]
                        ),
                        coverages=(
                            state.coverages
                            + [coverage]
                        ),
                        combinations=(
                            new_combinations
                        ),
                        distribution=(
                            new_distribution
                        ),
                        score=new_score,
                    )
                )

        if not next_states:
            raise ValueError(
                "No valid Beam Search states remain."
            )

        next_states.sort(
            key=lambda state: (
                state.score,
                state.distribution[-1],
                state.combinations,
            ),
            reverse=True,
        )

        states = next_states[
            :beam_width
        ]

    if not states:
        raise ValueError(
            "No valid ticket plan found."
        )

    states.sort(
        key=lambda state: (
            state.score,
            state.distribution[-1],
            state.combinations,
        ),
        reverse=True,
    )

    return states[0]


def calculate_incremental_gains(
    row: pd.Series,
) -> tuple[float, float]:
    """
    Single→Doubleで追加される確率
    Double→Tripleで追加される確率
    """
    ranked = ranked_outcomes(row)

    single_to_double = ranked[1][1]
    double_to_triple = ranked[2][1]

    return (
        single_to_double,
        double_to_triple,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--budget",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--exact-weight",
        type=float,
        default=1.0,
        help=(
            "全試合的中確率の重み。"
            "13試合なら1等相当。"
        ),
    )

    parser.add_argument(
        "--one-miss-weight",
        type=float,
        default=0.20,
        help=(
            "1試合外れ確率の重み。"
            "13試合なら2等相当。"
        ),
    )

    parser.add_argument(
        "--two-miss-weight",
        type=float,
        default=0.03,
        help=(
            "2試合外れ確率の重み。"
            "13試合なら3等相当。"
        ),
    )

    args = parser.parse_args()

    if args.budget < UNIT_PRICE:
        raise ValueError(
            "Budget must be at least 100 yen."
        )

    if args.beam_width < 1:
        raise ValueError(
            "Beam width must be at least 1."
        )

    weights = [
        args.exact_weight,
        args.one_miss_weight,
        args.two_miss_weight,
    ]

    if any(
        weight < 0
        for weight in weights
    ):
        raise ValueError(
            "Prize weights cannot be negative."
        )

    if sum(weights) <= 0:
        raise ValueError(
            "At least one prize weight "
            "must be positive."
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    required = {
        "homeTeam",
        "awayTeam",
        "prediction",
        "probH",
        "probD",
        "probA",
        "confidence",
        "margin",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    max_tickets = max(
        1,
        args.budget // UNIT_PRICE,
    )

    best_state = beam_search(
        df=df,
        max_tickets=max_tickets,
        beam_width=args.beam_width,
        exact_weight=args.exact_weight,
        one_miss_weight=(
            args.one_miss_weight
        ),
        two_miss_weight=(
            args.two_miss_weight
        ),
    )

    rows = []

    for size, (_, row) in zip(
        best_state.sizes,
        df.iterrows(),
    ):
        (
            ticket,
            ticket_type,
            coverage,
        ) = ticket_from_size(
            row,
            size,
        )

        ranked = ranked_outcomes(row)

        (
            single_to_double_gain,
            double_to_triple_gain,
        ) = calculate_incremental_gains(
            row
        )

        output_row = {
            "roundNo": row.get(
                "roundNo",
                "",
            ),
            "homeTeam": row["homeTeam"],
            "awayTeam": row["awayTeam"],
            "model": row.get(
                "model",
                "",
            ),
            "prediction": (
                row["prediction"]
            ),
            "actual": row.get(
                "actual",
                "",
            ),
            "probH": row["probH"],
            "probD": row["probD"],
            "probA": row["probA"],
            "confidence": (
                row["confidence"]
            ),
            "margin": row["margin"],
            "entropy": row.get(
                "entropy",
                "",
            ),
            "difficulty": row.get(
                "difficulty",
                "",
            ),
            "difficultyClass": row.get(
                "difficultyClass",
                "",
            ),
            "difficultyStars": row.get(
                "difficultyStars",
                "",
            ),
            "ticketType": ticket_type,
            "ticket": ticket,
            "coverage": coverage,
            "singleToDoubleGain": (
                single_to_double_gain
            ),
            "doubleToTripleGain": (
                double_to_triple_gain
            ),
            "top1": ranked[0][0],
            "top1Prob": ranked[0][1],
            "top2": ranked[1][0],
            "top2Prob": ranked[1][1],
            "top3": ranked[2][0],
            "top3Prob": ranked[2][1],
        }

        if (
            "drawSpecialistProb"
            in row.index
        ):
            output_row[
                "drawSpecialistProb"
            ] = row[
                "drawSpecialistProb"
            ]

        if (
            "dynamicAlpha"
            in row.index
        ):
            output_row[
                "dynamicAlpha"
            ] = row[
                "dynamicAlpha"
            ]

        rows.append(output_row)

    out = pd.DataFrame(rows)

    combinations = (
        best_state.combinations
    )

    cost = (
        combinations
        * UNIT_PRICE
    )

    distribution = (
        best_state.distribution
    )

    matches = len(df)

    all_correct = (
        distribution[matches]
    )

    one_miss = (
        distribution[matches - 1]
        if matches >= 1
        else 0.0
    )

    two_miss = (
        distribution[matches - 2]
        if matches >= 2
        else 0.0
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 56)
    print(
        "Budget Optimizer Prize Beam Search"
    )
    print("=" * 56)

    print(
        f"Budget             : "
        f"{args.budget:,} yen"
    )

    print(
        f"Max Tickets        : "
        f"{max_tickets}"
    )

    print(
        f"Total Tickets      : "
        f"{combinations}"
    )

    print(
        f"Cost               : "
        f"{cost:,} yen"
    )

    print()
    print(
        f"Exact Weight       : "
        f"{args.exact_weight:.4f}"
    )

    print(
        f"One Miss Weight    : "
        f"{args.one_miss_weight:.4f}"
    )

    print(
        f"Two Miss Weight    : "
        f"{args.two_miss_weight:.4f}"
    )

    print()
    print(
        f"All Correct        : "
        f"{all_correct:.6%}"
    )

    print(
        f"Exactly One Miss   : "
        f"{one_miss:.6%}"
    )

    print(
        f"Exactly Two Misses : "
        f"{two_miss:.6%}"
    )

    print(
        f"Optimization Score : "
        f"{best_state.score:.8f}"
    )

    print(f"Saved: {OUTPUT_PATH}")
    print()

    display_columns = [
        "homeTeam",
        "awayTeam",
        "probH",
        "probD",
        "probA",
        "difficulty",
        "ticketType",
        "ticket",
        "coverage",
        "singleToDoubleGain",
        "doubleToTripleGain",
    ]

    print(
        out[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("Summary")

    print(
        out[
            "ticketType"
        ].value_counts().to_string()
    )


if __name__ == "__main__":
    main()