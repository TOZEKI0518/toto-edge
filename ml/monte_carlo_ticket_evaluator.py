from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = Path("ml/round_predictions.csv")
DEFAULT_PLAN = Path("ml/budget_ticket_plan_beam.csv")
DEFAULT_OUTPUT = Path("ml/monte_carlo_summary.json")

OUTCOMES = ("H", "D", "A")
OUTCOME_INDEX = {
    outcome: index
    for index, outcome in enumerate(OUTCOMES)
}

UNIT_PRICE = 100


def normalize_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        1e-9,
        None,
    )

    return probabilities / probabilities.sum(
        axis=-1,
        keepdims=True,
    )


def ticket_to_mask(ticket: str) -> np.ndarray:
    ticket = str(ticket).upper()

    mask = np.array(
        [
            "H" in ticket,
            "D" in ticket,
            "A" in ticket,
        ],
        dtype=bool,
    )

    if not mask.any():
        raise ValueError(
            f"Invalid ticket value: {ticket}"
        )

    return mask


def ticket_count_from_plan(
    plan: pd.DataFrame,
) -> int:
    total = 1

    for ticket in plan["ticket"]:
        size = int(
            ticket_to_mask(ticket).sum()
        )

        total *= size

    return total


def load_and_align_data(
    predictions_path: Path,
    plan_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}"
        )

    if not plan_path.exists():
        raise FileNotFoundError(
            f"Ticket plan not found: {plan_path}"
        )

    predictions = pd.read_csv(
        predictions_path
    )

    plan = pd.read_csv(
        plan_path
    )

    required_predictions = {
        "homeTeam",
        "awayTeam",
        "probH",
        "probD",
        "probA",
    }

    required_plan = {
        "homeTeam",
        "awayTeam",
        "ticket",
    }

    missing_predictions = (
        required_predictions
        - set(predictions.columns)
    )

    missing_plan = (
        required_plan
        - set(plan.columns)
    )

    if missing_predictions:
        raise ValueError(
            "Missing prediction columns: "
            f"{sorted(missing_predictions)}"
        )

    if missing_plan:
        raise ValueError(
            "Missing ticket plan columns: "
            f"{sorted(missing_plan)}"
        )

    key_columns = [
        "homeTeam",
        "awayTeam",
    ]

    prediction_keys = list(
        predictions[key_columns].itertuples(
            index=False,
            name=None,
        )
    )

    plan_keys = list(
        plan[key_columns].itertuples(
            index=False,
            name=None,
        )
    )

    if prediction_keys != plan_keys:
        raise ValueError(
            "The match order differs between "
            "round_predictions.csv and the ticket plan."
        )

    return predictions, plan


def sample_uncertain_probabilities(
    base_probabilities: np.ndarray,
    simulations: int,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    戻り値:
        shape = (
            simulations,
            matches,
            3,
        )

    concentrationが大きい:
        予測確率の周辺に集中し、揺れが小さい

    concentrationが小さい:
        確率不確実性が大きい
    """
    if concentration <= 0:
        raise ValueError(
            "Concentration must be greater than zero."
        )

    simulations_count = simulations
    matches = len(base_probabilities)

    sampled = np.empty(
        (
            simulations_count,
            matches,
            3,
        ),
        dtype=np.float32,
    )

    for match_index, probabilities in enumerate(
        base_probabilities
    ):
        alpha = (
            probabilities
            * concentration
        )

        alpha = np.clip(
            alpha,
            1e-6,
            None,
        )

        sampled[
            :,
            match_index,
            :,
        ] = rng.dirichlet(
            alpha=alpha,
            size=simulations_count,
        )

    return sampled


def sample_results(
    uncertain_probabilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    H=0, D=1, A=2として結果を生成する。

    戻り値:
        shape = (
            simulations,
            matches,
        )
    """
    random_values = rng.random(
        uncertain_probabilities.shape[:2]
    )

    cumulative_h = (
        uncertain_probabilities[:, :, 0]
    )

    cumulative_d = (
        cumulative_h
        + uncertain_probabilities[:, :, 1]
    )

    results = np.full(
        random_values.shape,
        2,
        dtype=np.int8,
    )

    results[
        random_values < cumulative_d
    ] = 1

    results[
        random_values < cumulative_h
    ] = 0

    return results


def evaluate_plan_against_results(
    plan: pd.DataFrame,
    results: np.ndarray,
) -> np.ndarray:
    ticket_masks = np.stack(
        [
            ticket_to_mask(ticket)
            for ticket in plan["ticket"]
        ],
        axis=0,
    )

    simulations, matches = results.shape

    covered = np.empty(
        (
            simulations,
            matches,
        ),
        dtype=bool,
    )

    for match_index in range(matches):
        covered[:, match_index] = (
            ticket_masks[
                match_index,
                results[:, match_index],
            ]
        )

    return covered.sum(axis=1)


def calculate_distribution(
    hit_counts: np.ndarray,
    matches: int,
) -> dict[str, float]:
    counts = np.bincount(
        hit_counts,
        minlength=matches + 1,
    )

    probabilities = (
        counts / len(hit_counts)
    )

    return {
        str(hits): float(
            probabilities[hits]
        )
        for hits in range(matches + 1)
    }


def summarize_run(
    hit_counts: np.ndarray,
    ticket_count: int,
    concentration: float,
) -> dict:
    matches = int(
        hit_counts.max(
            initial=0
        )
    )

    # 13試合でない場合にも対応できるよう、
    # 最大的中数ではなく呼び出し側で補正する
    return {
        "ticketCount": ticket_count,
        "investment": (
            ticket_count * UNIT_PRICE
        ),
        "concentration": concentration,
        "meanHits": float(
            hit_counts.mean()
        ),
        "stdHits": float(
            hit_counts.std()
        ),
    }


def evaluate_ticket_plan(
    predictions: pd.DataFrame,
    plan: pd.DataFrame,
    simulations: int,
    concentration: float,
    seed: int,
    batch_size: int,
) -> dict:
    if simulations < 1:
        raise ValueError(
            "Simulations must be at least 1."
        )

    if batch_size < 1:
        raise ValueError(
            "Batch size must be at least 1."
        )

    base_probabilities = normalize_probabilities(
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

    matches = len(predictions)
    ticket_count = ticket_count_from_plan(
        plan
    )

    rng = np.random.default_rng(
        seed
    )

    all_hit_counts = []

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
                simulations=current_batch,
                concentration=concentration,
                rng=rng,
            )
        )

        sampled_results = sample_results(
            uncertain_probabilities=(
                uncertain_probabilities
            ),
            rng=rng,
        )

        hit_counts = (
            evaluate_plan_against_results(
                plan=plan,
                results=sampled_results,
            )
        )

        all_hit_counts.append(
            hit_counts
        )

        remaining -= current_batch

    hit_counts = np.concatenate(
        all_hit_counts
    )

    distribution = calculate_distribution(
        hit_counts=hit_counts,
        matches=matches,
    )

    exact_probability = float(
        (hit_counts == matches).mean()
    )

    one_miss_probability = float(
        (hit_counts == matches - 1).mean()
    ) if matches >= 1 else 0.0

    two_miss_probability = float(
        (hit_counts == matches - 2).mean()
    ) if matches >= 2 else 0.0

    return {
        "simulations": simulations,
        "seed": seed,
        "matches": matches,
        "concentration": concentration,
        "ticketCount": ticket_count,
        "investment": (
            ticket_count * UNIT_PRICE
        ),
        "meanHits": float(
            hit_counts.mean()
        ),
        "stdHits": float(
            hit_counts.std()
        ),
        "p05Hits": float(
            np.quantile(
                hit_counts,
                0.05,
            )
        ),
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
        "exactProbability": (
            exact_probability
        ),
        "oneMissProbability": (
            one_miss_probability
        ),
        "twoMissProbability": (
            two_miss_probability
        ),
        "oneMissOrBetterProbability": float(
            exact_probability
            + one_miss_probability
        ),
        "twoMissOrBetterProbability": float(
            exact_probability
            + one_miss_probability
            + two_miss_probability
        ),
        "hitDistribution": distribution,
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
        "--plan",
        default=str(
            DEFAULT_PLAN
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--concentration",
        type=float,
        default=50.0,
        help=(
            "Dirichlet concentration。"
            "大きいほど確率の揺れが小さい。"
        ),
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

    plan_path = Path(
        args.plan
    )

    output_path = Path(
        args.output
    )

    predictions, plan = (
        load_and_align_data(
            predictions_path=(
                predictions_path
            ),
            plan_path=plan_path,
        )
    )

    summary = evaluate_ticket_plan(
        predictions=predictions,
        plan=plan,
        simulations=args.simulations,
        concentration=(
            args.concentration
        ),
        seed=args.seed,
        batch_size=args.batch_size,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    matches = summary["matches"]

    print("=" * 60)
    print("Monte Carlo Ticket Evaluator")
    print("=" * 60)

    print(
        f"Simulations       : "
        f"{summary['simulations']:,}"
    )

    print(
        f"Matches           : "
        f"{matches}"
    )

    print(
        f"Ticket Count      : "
        f"{summary['ticketCount']:,}"
    )

    print(
        f"Investment        : "
        f"{summary['investment']:,} yen"
    )

    print(
        f"Concentration     : "
        f"{summary['concentration']:.2f}"
    )

    print()
    print(
        f"Mean Hits         : "
        f"{summary['meanHits']:.4f}"
    )

    print(
        f"Std Hits          : "
        f"{summary['stdHits']:.4f}"
    )

    print(
        f"P05 / P50 / P95   : "
        f"{summary['p05Hits']:.0f} / "
        f"{summary['p50Hits']:.0f} / "
        f"{summary['p95Hits']:.0f}"
    )

    print()
    print(
        f"All Correct       : "
        f"{summary['exactProbability']:.6%}"
    )

    print(
        f"1 Miss Exactly    : "
        f"{summary['oneMissProbability']:.6%}"
    )

    print(
        f"2 Misses Exactly  : "
        f"{summary['twoMissProbability']:.6%}"
    )

    print(
        f"1 Miss or Better  : "
        f"{summary['oneMissOrBetterProbability']:.6%}"
    )

    print(
        f"2 Miss or Better  : "
        f"{summary['twoMissOrBetterProbability']:.6%}"
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()