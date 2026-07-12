from pathlib import Path
import argparse

import pandas as pd

from common.match_difficulty import (
    MatchDifficulty,
)


INPUT_PATH = Path(
    "ml/round_predictions.csv"
)

OUTPUT_PATH = Path(
    "ml/budget_ticket_plan_beam.csv"
)

UNIT_PRICE = 100


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


def get_difficulty(
    row: pd.Series,
    engine: MatchDifficulty,
) -> float:
    if (
        "difficulty" in row.index
        and pd.notna(row["difficulty"])
    ):
        return float(row["difficulty"])

    result = engine.calculate(
        prob_h=float(row["probH"]),
        prob_d=float(row["probD"]),
        prob_a=float(row["probA"]),
    )

    return result.difficulty


def difficulty_multiplier(
    difficulty: float,
    ticket_size: int,
    difficulty_bonus: float,
) -> float:
    """
    Single  : 難易度ボーナスなし
    Double  : 難易度ボーナスの50%
    Triple  : 難易度ボーナスの100%

    難しい試合にDouble/Tripleを割り当てた候補を
    Beam Search内で少し優遇する。
    """
    expansion_ratio = (
        ticket_size - 1
    ) / 2.0

    return (
        1.0
        + difficulty_bonus
        * difficulty
        * expansion_ratio
    )


def beam_search(
    df: pd.DataFrame,
    max_tickets: int,
    beam_width: int,
    difficulty_bonus: float,
) -> tuple[
    list[int],
    int,
    float,
    float,
]:
    """
    state:
        sizes
        combinations
        raw_coverage
        adjusted_score
    """

    difficulty_engine = (
        MatchDifficulty()
    )

    states = [
        (
            [],
            1,
            1.0,
            1.0,
        )
    ]

    for _, row in df.iterrows():
        next_states = []

        difficulty = get_difficulty(
            row,
            difficulty_engine,
        )

        for (
            sizes,
            combinations,
            raw_coverage,
            adjusted_score,
        ) in states:
            for size in (1, 2, 3):
                new_combinations = (
                    combinations * size
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

                multiplier = (
                    difficulty_multiplier(
                        difficulty=difficulty,
                        ticket_size=size,
                        difficulty_bonus=(
                            difficulty_bonus
                        ),
                    )
                )

                next_states.append(
                    (
                        sizes + [size],
                        new_combinations,
                        raw_coverage
                        * coverage,
                        adjusted_score
                        * coverage
                        * multiplier,
                    )
                )

        if not next_states:
            raise ValueError(
                "No valid Beam Search "
                "states remain."
            )

        next_states.sort(
            key=lambda state: (
                state[3],
                state[2],
                state[1],
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
            state[3],
            state[2],
            state[1],
        ),
        reverse=True,
    )

    return states[0]


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
        default=5000,
    )

    parser.add_argument(
        "--difficulty-bonus",
        type=float,
        default=0.30,
        help=(
            "難しい試合へDouble/Tripleを"
            "割り当てる優遇強度。"
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

    if args.difficulty_bonus < 0:
        raise ValueError(
            "Difficulty bonus cannot be negative."
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

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

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    difficulty_engine = (
        MatchDifficulty()
    )

    if "difficulty" not in df.columns:
        df["difficulty"] = df.apply(
            lambda row: get_difficulty(
                row,
                difficulty_engine,
            ),
            axis=1,
        )

    if "difficultyClass" not in df.columns:
        difficulty_results = df.apply(
            lambda row: (
                difficulty_engine.calculate(
                    prob_h=float(
                        row["probH"]
                    ),
                    prob_d=float(
                        row["probD"]
                    ),
                    prob_a=float(
                        row["probA"]
                    ),
                )
            ),
            axis=1,
        )

        df["difficultyClass"] = [
            result.difficulty_class
            for result in difficulty_results
        ]

        df["difficultyStars"] = [
            result.difficulty_stars
            for result in difficulty_results
        ]

    max_tickets = max(
        1,
        args.budget // UNIT_PRICE,
    )

    (
        sizes,
        combinations,
        raw_coverage,
        adjusted_score,
    ) = beam_search(
        df=df,
        max_tickets=max_tickets,
        beam_width=args.beam_width,
        difficulty_bonus=(
            args.difficulty_bonus
        ),
    )

    rows = []

    for size, (_, row) in zip(
        sizes,
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

        multiplier = (
            difficulty_multiplier(
                difficulty=float(
                    row["difficulty"]
                ),
                ticket_size=size,
                difficulty_bonus=(
                    args.difficulty_bonus
                ),
            )
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
            "difficulty": (
                row["difficulty"]
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
            "difficultyMultiplier": (
                multiplier
            ),
            "beamMatchScore": (
                coverage * multiplier
            ),
            "top1": ranked[0][0],
            "top1Prob": ranked[0][1],
            "top2": ranked[1][0],
            "top2Prob": ranked[1][1],
            "top3": ranked[2][0],
            "top3Prob": ranked[2][1],
        }

        if "drawSpecialistProb" in row.index:
            output_row[
                "drawSpecialistProb"
            ] = row[
                "drawSpecialistProb"
            ]

        if "dynamicAlpha" in row.index:
            output_row[
                "dynamicAlpha"
            ] = row["dynamicAlpha"]

        rows.append(output_row)

    out = pd.DataFrame(rows)

    cost = (
        combinations * UNIT_PRICE
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

    print("=" * 48)
    print(
        "Budget Optimizer "
        "Difficulty Beam Search"
    )
    print("=" * 48)

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

    print(
        f"Difficulty Bonus   : "
        f"{args.difficulty_bonus:.2f}"
    )

    print(
        f"Raw Coverage       : "
        f"{raw_coverage:.6f}"
    )

    print(
        f"Adjusted Beam Score: "
        f"{adjusted_score:.6f}"
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
        "difficultyStars",
        "ticketType",
        "ticket",
        "coverage",
        "beamMatchScore",
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