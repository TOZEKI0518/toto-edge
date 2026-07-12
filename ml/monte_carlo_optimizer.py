from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from monte_carlo_ticket_evaluator import (
    OUTCOMES,
    UNIT_PRICE,
    evaluate_plan_against_results,
    normalize_probabilities,
    sample_results,
    sample_uncertain_probabilities,
    ticket_count_from_plan,
)


DEFAULT_PREDICTIONS = Path(
    "ml/round_predictions.csv"
)

DEFAULT_OUTPUT_PLAN = Path(
    "ml/budget_ticket_plan_monte_carlo.csv"
)

DEFAULT_OUTPUT_SUMMARY = Path(
    "ml/monte_carlo_summary.json"
)


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
            ranked[0][0]
            + ranked[1][0],
            "Double",
            ranked[0][1]
            + ranked[1][1],
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


def combinations_from_sizes(
    sizes: list[int],
) -> int:
    return math.prod(sizes)


def build_plan_from_sizes(
    predictions: pd.DataFrame,
    sizes: list[int],
) -> pd.DataFrame:
    rows = []

    for size, (_, row) in zip(
        sizes,
        predictions.iterrows(),
    ):
        (
            ticket,
            ticket_type,
            coverage,
        ) = ticket_from_size(
            row,
            size,
        )

        ranked = ranked_outcomes(
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
            "prediction": row.get(
                "prediction",
                ranked[0][0],
            ),
            "actual": row.get(
                "actual",
                "",
            ),
            "probH": row["probH"],
            "probD": row["probD"],
            "probA": row["probA"],
            "confidence": row.get(
                "confidence",
                ranked[0][1],
            ),
            "margin": row.get(
                "margin",
                ranked[0][1]
                - ranked[1][1],
            ),
            "difficulty": row.get(
                "difficulty",
                "",
            ),
            "ticketType": ticket_type,
            "ticket": ticket,
            "coverage": coverage,
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
            ] = row[
                "dynamicAlpha"
            ]

        rows.append(
            output_row
        )

    return pd.DataFrame(rows)


def initial_candidate(
    predictions: pd.DataFrame,
    max_tickets: int,
) -> list[int]:
    """
    全Singleから開始し、追加確率が大きい順に
    Single→Double、Double→Tripleへ拡張する。
    """
    sizes = [1] * len(predictions)

    while True:
        current_combinations = (
            combinations_from_sizes(sizes)
        )

        best_action = None
        best_efficiency = -1.0

        for match_index, (
            size,
            (_, row),
        ) in enumerate(
            zip(
                sizes,
                predictions.iterrows(),
            )
        ):
            if size >= 3:
                continue

            new_size = size + 1

            new_combinations = (
                current_combinations
                // size
                * new_size
            )

            if new_combinations > max_tickets:
                continue

            ranked = ranked_outcomes(
                row
            )

            added_probability = (
                ranked[size][1]
            )

            extra_tickets = (
                new_combinations
                - current_combinations
            )

            efficiency = (
                added_probability
                / extra_tickets
            )

            if efficiency > best_efficiency:
                best_efficiency = (
                    efficiency
                )
                best_action = (
                    match_index,
                    new_size,
                )

        if best_action is None:
            break

        match_index, new_size = (
            best_action
        )

        sizes[match_index] = (
            new_size
        )

    return sizes


def mutate_candidate(
    sizes: list[int],
    max_tickets: int,
    rng: np.random.Generator,
) -> list[int]:
    candidate = sizes.copy()

    expanded_indexes = [
        index
        for index, size
        in enumerate(candidate)
        if size > 1
    ]

    reducible_indexes = [
        index
        for index, size
        in enumerate(candidate)
        if size < 3
    ]

    if not expanded_indexes:
        return candidate

    remove_index = int(
        rng.choice(
            expanded_indexes
        )
    )

    candidate[remove_index] -= 1

    rng.shuffle(
        reducible_indexes
    )

    for add_index in reducible_indexes:
        if add_index == remove_index:
            continue

        candidate[add_index] += 1

        if (
            combinations_from_sizes(
                candidate
            )
            <= max_tickets
        ):
            return candidate

        candidate[add_index] -= 1

    return candidate


def generate_candidates(
    predictions: pd.DataFrame,
    max_tickets: int,
    candidate_count: int,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(
        seed
    )

    initial = initial_candidate(
        predictions=predictions,
        max_tickets=max_tickets,
    )

    candidates = {
        tuple(initial)
    }

    attempts = 0
    maximum_attempts = (
        candidate_count * 100
    )

    current = initial

    while (
        len(candidates) < candidate_count
        and attempts < maximum_attempts
    ):
        mutated = mutate_candidate(
            sizes=current,
            max_tickets=max_tickets,
            rng=rng,
        )

        if (
            combinations_from_sizes(mutated)
            <= max_tickets
        ):
            candidates.add(
                tuple(mutated)
            )
            current = mutated

        attempts += 1

    return [
        list(candidate)
        for candidate in candidates
    ]


def create_simulation_results(
    predictions: pd.DataFrame,
    simulations: int,
    concentration: float,
    seed: int,
    batch_size: int,
) -> list[np.ndarray]:
    base_probabilities = (
        normalize_probabilities(
            predictions[
                [
                    "probH",
                    "probD",
                    "probA",
                ]
            ].to_numpy(
                dtype=float
            )
        )
    )

    rng = np.random.default_rng(
        seed
    )

    result_batches = []

    remaining = simulations

    while remaining > 0:
        current_batch = min(
            batch_size,
            remaining,
        )

        uncertain_probabilities = (
            sample_uncertain_probabilities(
                base_probabilities=(
                    base_probabilities
                ),
                simulations=(
                    current_batch
                ),
                concentration=(
                    concentration
                ),
                rng=rng,
            )
        )

        sampled_results = sample_results(
            uncertain_probabilities=(
                uncertain_probabilities
            ),
            rng=rng,
        )

        result_batches.append(
            sampled_results
        )

        remaining -= current_batch

    return result_batches


def evaluate_candidate(
    plan: pd.DataFrame,
    result_batches: list[np.ndarray],
    matches: int,
    exact_weight: float,
    one_miss_weight: float,
    two_miss_weight: float,
    robustness_weight: float,
) -> dict:
    all_hit_counts = []

    for results in result_batches:
        hit_counts = (
            evaluate_plan_against_results(
                plan=plan,
                results=results,
            )
        )

        all_hit_counts.append(
            hit_counts
        )

    hit_counts = np.concatenate(
        all_hit_counts
    )

    exact = float(
        (hit_counts == matches).mean()
    )

    one_miss = float(
        (hit_counts == matches - 1).mean()
    )

    two_miss = float(
        (hit_counts == matches - 2).mean()
    )

    mean_hits = float(
        hit_counts.mean()
    )

    lower_tail_hits = float(
        np.quantile(
            hit_counts,
            0.05,
        )
    )

    expected_score = (
        exact_weight * exact
        + one_miss_weight * one_miss
        + two_miss_weight * two_miss
    )

    normalized_lower_tail = (
        lower_tail_hits / matches
    )

    robustness_score = (
        expected_score
        + robustness_weight
        * normalized_lower_tail
    )

    return {
        "exactProbability": exact,
        "oneMissProbability": (
            one_miss
        ),
        "twoMissProbability": (
            two_miss
        ),
        "oneMissOrBetterProbability": (
            exact + one_miss
        ),
        "twoMissOrBetterProbability": (
            exact
            + one_miss
            + two_miss
        ),
        "meanHits": mean_hits,
        "p05Hits": lower_tail_hits,
        "p50Hits": float(
            np.quantile(
                hit_counts,
                0.50,
            )
        ),
        "p95Hits": float(
            np.quantile(
                hit_counts,
                0.95,
            )
        ),
        "expectedScore": (
            expected_score
        ),
        "robustnessScore": (
            robustness_score
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        default=str(
            DEFAULT_PREDICTIONS
        ),
    )

    parser.add_argument(
        "--output-plan",
        default=str(
            DEFAULT_OUTPUT_PLAN
        ),
    )

    parser.add_argument(
        "--output-summary",
        default=str(
            DEFAULT_OUTPUT_SUMMARY
        ),
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--concentration",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--exact-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--one-miss-weight",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--two-miss-weight",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--robustness-weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
    )

    args = parser.parse_args()

    predictions_path = Path(
        args.predictions
    )

    output_plan_path = Path(
        args.output_plan
    )

    output_summary_path = Path(
        args.output_summary
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: "
            f"{predictions_path}"
        )

    predictions = pd.read_csv(
        predictions_path
    )

    required = {
        "homeTeam",
        "awayTeam",
        "probH",
        "probD",
        "probA",
    }

    missing = required - set(
        predictions.columns
    )

    if missing:
        raise ValueError(
            f"Missing prediction columns: "
            f"{sorted(missing)}"
        )

    if args.budget < UNIT_PRICE:
        raise ValueError(
            "Budget must be at least 100 yen."
        )

    if args.candidate_count < 1:
        raise ValueError(
            "Candidate count must be at least 1."
        )

    max_tickets = (
        args.budget // UNIT_PRICE
    )

    candidates = generate_candidates(
        predictions=predictions,
        max_tickets=max_tickets,
        candidate_count=(
            args.candidate_count
        ),
        seed=args.seed,
    )

    result_batches = (
        create_simulation_results(
            predictions=predictions,
            simulations=args.simulations,
            concentration=(
                args.concentration
            ),
            seed=args.seed,
            batch_size=args.batch_size,
        )
    )

    matches = len(predictions)

    candidate_results = []

    print("=" * 64)
    print("Monte Carlo Ticket Optimizer")
    print("=" * 64)

    print(
        f"Matches           : "
        f"{matches}"
    )

    print(
        f"Budget            : "
        f"{args.budget:,} yen"
    )

    print(
        f"Max Tickets       : "
        f"{max_tickets}"
    )

    print(
        f"Candidates        : "
        f"{len(candidates):,}"
    )

    print(
        f"Simulations       : "
        f"{args.simulations:,}"
    )

    print(
        f"Concentration     : "
        f"{args.concentration:.2f}"
    )

    print()

    for candidate_number, sizes in enumerate(
        candidates,
        start=1,
    ):
        plan = build_plan_from_sizes(
            predictions=predictions,
            sizes=sizes,
        )

        metrics = evaluate_candidate(
            plan=plan,
            result_batches=result_batches,
            matches=matches,
            exact_weight=args.exact_weight,
            one_miss_weight=(
                args.one_miss_weight
            ),
            two_miss_weight=(
                args.two_miss_weight
            ),
            robustness_weight=(
                args.robustness_weight
            ),
        )

        combinations = (
            combinations_from_sizes(
                sizes
            )
        )

        candidate_results.append(
            {
                "candidateNumber": (
                    candidate_number
                ),
                "sizes": sizes,
                "ticketCount": (
                    combinations
                ),
                "investment": (
                    combinations
                    * UNIT_PRICE
                ),
                "plan": plan,
                **metrics,
            }
        )

    candidate_results.sort(
        key=lambda result: (
            result["robustnessScore"],
            result[
                "twoMissOrBetterProbability"
            ],
            result[
                "oneMissOrBetterProbability"
            ],
            result["exactProbability"],
        ),
        reverse=True,
    )

    best = candidate_results[0]

    best_plan = best["plan"]

    output_plan_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_plan.to_csv(
        output_plan_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "method": (
            "dirichlet_uncertainty_"
            "monte_carlo_optimizer"
        ),
        "simulations": (
            args.simulations
        ),
        "candidateCount": (
            len(candidates)
        ),
        "matches": matches,
        "budgetLimit": (
            args.budget
        ),
        "ticketCount": (
            best["ticketCount"]
        ),
        "investment": (
            best["investment"]
        ),
        "concentration": (
            args.concentration
        ),
        "seed": args.seed,
        "weights": {
            "exact": (
                args.exact_weight
            ),
            "oneMiss": (
                args.one_miss_weight
            ),
            "twoMiss": (
                args.two_miss_weight
            ),
            "robustness": (
                args.robustness_weight
            ),
        },
        "exactProbability": (
            best["exactProbability"]
        ),
        "oneMissProbability": (
            best["oneMissProbability"]
        ),
        "twoMissProbability": (
            best["twoMissProbability"]
        ),
        "oneMissOrBetterProbability": (
            best[
                "oneMissOrBetterProbability"
            ]
        ),
        "twoMissOrBetterProbability": (
            best[
                "twoMissOrBetterProbability"
            ]
        ),
        "meanHits": best["meanHits"],
        "p05Hits": best["p05Hits"],
        "p50Hits": best["p50Hits"],
        "p95Hits": best["p95Hits"],
        "expectedScore": (
            best["expectedScore"]
        ),
        "robustnessScore": (
            best["robustnessScore"]
        ),
        "ticketSizes": best["sizes"],
    }

    output_summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Selected Tickets  : "
        f"{best['ticketCount']}"
    )

    print(
        f"Investment        : "
        f"{best['investment']:,} yen"
    )

    print()
    print(
        f"All Correct       : "
        f"{best['exactProbability']:.6%}"
    )

    print(
        f"1 Miss Exactly    : "
        f"{best['oneMissProbability']:.6%}"
    )

    print(
        f"2 Misses Exactly  : "
        f"{best['twoMissProbability']:.6%}"
    )

    print(
        f"1 Miss or Better  : "
        f"{best['oneMissOrBetterProbability']:.6%}"
    )

    print(
        f"2 Miss or Better  : "
        f"{best['twoMissOrBetterProbability']:.6%}"
    )

    print()
    print(
        f"Mean Hits         : "
        f"{best['meanHits']:.4f}"
    )

    print(
        f"P05 / P50 / P95   : "
        f"{best['p05Hits']:.0f} / "
        f"{best['p50Hits']:.0f} / "
        f"{best['p95Hits']:.0f}"
    )

    print(
        f"Robustness Score  : "
        f"{best['robustnessScore']:.8f}"
    )

    print()
    print(
        best_plan[
            [
                "homeTeam",
                "awayTeam",
                "probH",
                "probD",
                "probA",
                "ticketType",
                "ticket",
                "coverage",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved plan    : "
        f"{output_plan_path}"
    )

    print(
        f"Saved summary : "
        f"{output_summary_path}"
    )


if __name__ == "__main__":
    main()