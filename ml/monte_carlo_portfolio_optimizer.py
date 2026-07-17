from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Iterable

import numpy as np
import pandas as pd

LOGGER: Final[logging.Logger] = logging.getLogger(
    "monte_carlo_portfolio_optimizer"
)

OUTCOMES: Final[tuple[str, str, str]] = ("A", "D", "H")
AI_COLUMNS: Final[dict[str, str]] = {
    "A": "prob_away",
    "D": "prob_draw",
    "H": "prob_home",
}
MARKET_COLUMNS: Final[dict[str, str]] = {
    "A": "market_prob_away",
    "D": "market_prob_draw",
    "H": "market_prob_home",
}


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    """Configuration for portfolio-level Monte Carlo optimization."""

    confidence_csv: Path
    candidate_csv: Path
    output_dir: Path
    ticket_count: int = 50
    simulations: int = 100_000
    optimization_iterations: int = 10_000
    replacement_pool_size: int = 1_000
    seed: int = 42
    model_weight: float = 0.70
    market_weight: float = 0.30
    hit_rate_weight: float = 0.55
    diversity_weight: float = 0.20
    value_weight: float = 0.20
    tail_weight: float = 0.05
    minimum_unique_ratio: float = 0.80
    maximum_pair_similarity: float = 0.92
    ticket_price_yen: int = 100

    def validate(self) -> None:
        """Validate configuration values and required files."""
        if not self.confidence_csv.exists():
            raise FileNotFoundError(
                f"Confidence CSV not found: {self.confidence_csv}"
            )
        if not self.candidate_csv.exists():
            raise FileNotFoundError(
                f"Candidate CSV not found: {self.candidate_csv}"
            )
        if self.ticket_count <= 0:
            raise ValueError("ticket_count must be positive.")
        if self.simulations <= 0:
            raise ValueError("simulations must be positive.")
        if self.optimization_iterations <= 0:
            raise ValueError(
                "optimization_iterations must be positive."
            )
        if self.replacement_pool_size < self.ticket_count:
            raise ValueError(
                "replacement_pool_size must be >= ticket_count."
            )
        if self.ticket_price_yen <= 0:
            raise ValueError("ticket_price_yen must be positive.")
        if not math.isclose(
            self.model_weight + self.market_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "model_weight + market_weight must equal 1.0."
            )


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    """Metrics calculated for one candidate ticket portfolio."""

    ticket_count: int
    exact_hit_probability: float
    simulated_hit_rate: float
    mean_best_match_count: float
    probability_10_or_more: float
    probability_11_or_more: float
    probability_12_or_more: float
    probability_13: float
    average_pair_similarity: float
    maximum_pair_similarity: float
    unique_ticket_ratio: float
    average_value_index: float
    tail_coverage_score: float
    objective_score: float


@dataclass(frozen=True, slots=True)
class OptimizationSummary:
    """Before/after optimization summary."""

    round_id: int
    seed: int
    simulations: int
    optimization_iterations: int
    ticket_count: int
    investment_yen: int
    initial_metrics: PortfolioMetrics
    optimized_metrics: PortfolioMetrics
    objective_improvement: float
    hit_rate_improvement: float
    probability_11_plus_improvement: float
    probability_12_plus_improvement: float


class MonteCarloOptimizerError(RuntimeError):
    """Raised when Monte Carlo optimization cannot run safely."""


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def project_root() -> Path:
    """Return project root when this file is saved under ml/."""
    return Path(__file__).resolve().parents[1]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Normalize probability rows."""
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise MonteCarloOptimizerError(
            "Probability rows must have positive sums."
        )
    return values / totals


def load_confidence_data(path: Path) -> pd.DataFrame:
    """Load and validate the 13-match confidence output."""
    frame = pd.read_csv(path, encoding="utf-8-sig")

    required = {
        "round_id",
        "toto_match_no",
        "match_card_id",
        *AI_COLUMNS.values(),
        *MARKET_COLUMNS.values(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MonteCarloOptimizerError(
            "Confidence CSV missing required columns: "
            + ", ".join(missing)
        )

    frame = frame.sort_values(
        "toto_match_no",
        kind="stable",
    ).reset_index(drop=True)

    if len(frame) != 13:
        raise MonteCarloOptimizerError(
            f"Expected 13 confidence rows, found {len(frame)}."
        )

    expected = list(range(1, 14))
    actual = frame["toto_match_no"].astype(int).tolist()
    if actual != expected:
        raise MonteCarloOptimizerError(
            f"Expected toto_match_no 1..13, found {actual}."
        )

    model = frame[
        [
            AI_COLUMNS["A"],
            AI_COLUMNS["D"],
            AI_COLUMNS["H"],
        ]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    market = frame[
        [
            MARKET_COLUMNS["A"],
            MARKET_COLUMNS["D"],
            MARKET_COLUMNS["H"],
        ]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(model).all():
        raise MonteCarloOptimizerError(
            "Model probabilities contain invalid values."
        )
    if not np.isfinite(market).all():
        raise MonteCarloOptimizerError(
            "Market probabilities contain invalid values."
        )
    if (model < 0.0).any() or (market < 0.0).any():
        raise MonteCarloOptimizerError(
            "Probabilities must not be negative."
        )

    frame.loc[
        :,
        [
            AI_COLUMNS["A"],
            AI_COLUMNS["D"],
            AI_COLUMNS["H"],
        ],
    ] = normalize_rows(model)

    frame.loc[
        :,
        [
            MARKET_COLUMNS["A"],
            MARKET_COLUMNS["D"],
            MARKET_COLUMNS["H"],
        ],
    ] = normalize_rows(market)

    return frame


def load_candidates(
    path: Path,
    *,
    minimum_count: int,
) -> pd.DataFrame:
    """Load and validate ticket candidates."""
    frame = pd.read_csv(path, encoding="utf-8-sig")

    required = {
        "picks",
        "model_probability",
        "market_probability",
        "conservative_value_index",
        "search_score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MonteCarloOptimizerError(
            "Candidate CSV missing required columns: "
            + ", ".join(missing)
        )

    frame["picks"] = frame["picks"].astype(str).str.strip().str.upper()
    valid_ticket = frame["picks"].str.fullmatch(r"[ADH]{13}")
    if not valid_ticket.all():
        invalid = frame.loc[~valid_ticket, "picks"].head(5).tolist()
        raise MonteCarloOptimizerError(
            f"Invalid candidate ticket strings: {invalid}"
        )

    for column in (
        "model_probability",
        "market_probability",
        "conservative_value_index",
        "search_score",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if frame[
        [
            "model_probability",
            "market_probability",
            "conservative_value_index",
            "search_score",
        ]
    ].isna().any().any():
        raise MonteCarloOptimizerError(
            "Candidate metrics contain missing or non-numeric values."
        )

    frame = (
        frame.sort_values(
            ["search_score", "model_probability"],
            ascending=[False, False],
            kind="stable",
        )
        .drop_duplicates(subset=["picks"], keep="first")
        .reset_index(drop=True)
    )

    if len(frame) < minimum_count:
        raise MonteCarloOptimizerError(
            f"Need at least {minimum_count} unique candidates, "
            f"found {len(frame)}."
        )

    return frame


def probability_tensor(
    frame: pd.DataFrame,
    config: MonteCarloConfig,
) -> np.ndarray:
    """
    Blend model and market probabilities for simulation.

    This preserves model alpha while reducing overconfidence through the
    observable market distribution.
    """
    model = frame[
        [
            AI_COLUMNS["A"],
            AI_COLUMNS["D"],
            AI_COLUMNS["H"],
        ]
    ].to_numpy(dtype=float)

    market = frame[
        [
            MARKET_COLUMNS["A"],
            MARKET_COLUMNS["D"],
            MARKET_COLUMNS["H"],
        ]
    ].to_numpy(dtype=float)

    blended = (
        config.model_weight * model
        + config.market_weight * market
    )
    return normalize_rows(blended)


def simulate_outcomes(
    probabilities: np.ndarray,
    *,
    simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate complete 13-match result vectors."""
    match_count = probabilities.shape[0]
    random_values = rng.random((simulations, match_count))
    cumulative = probabilities.cumsum(axis=1)

    outcomes = np.zeros(
        (simulations, match_count),
        dtype=np.int8,
    )
    outcomes[random_values > cumulative[:, 0]] = 1
    outcomes[random_values > cumulative[:, 1]] = 2
    return outcomes


def encode_ticket(ticket: str) -> np.ndarray:
    """Encode A/D/H ticket text as integer outcome indexes."""
    mapping = {"A": 0, "D": 1, "H": 2}
    return np.fromiter(
        (mapping[value] for value in ticket),
        dtype=np.int8,
        count=13,
    )


def encode_tickets(tickets: Iterable[str]) -> np.ndarray:
    """Encode multiple tickets."""
    rows = [encode_ticket(ticket) for ticket in tickets]
    return np.vstack(rows)


def pairwise_similarity_matrix(
    encoded_tickets: np.ndarray,
) -> np.ndarray:
    """Calculate pairwise matching-pick ratios."""
    return (
        encoded_tickets[:, None, :]
        == encoded_tickets[None, :, :]
    ).mean(axis=2)


def portfolio_best_match_counts(
    encoded_tickets: np.ndarray,
    simulations: np.ndarray,
    *,
    batch_size: int = 5_000,
) -> np.ndarray:
    """Return the best ticket match count for each simulated outcome."""
    best_counts = np.empty(len(simulations), dtype=np.int8)

    for start in range(0, len(simulations), batch_size):
        stop = min(start + batch_size, len(simulations))
        batch = simulations[start:stop]
        matches = (
            batch[:, None, :]
            == encoded_tickets[None, :, :]
        ).sum(axis=2)
        best_counts[start:stop] = matches.max(axis=1)

    return best_counts


def exact_portfolio_probability(
    tickets: pd.DataFrame,
) -> float:
    """
    Approximate exact-hit probability as the sum of unique ticket
    probabilities.

    Tickets are mutually exclusive exact 13-match result paths, so their
    probabilities may be added directly.
    """
    return float(tickets["model_probability"].sum())


def tail_coverage_score(best_counts: np.ndarray) -> float:
    """Score strong near-hit and exact-hit simulation outcomes."""
    return float(
        np.mean(
            np.select(
                [
                    best_counts == 13,
                    best_counts == 12,
                    best_counts == 11,
                    best_counts == 10,
                ],
                [1.0, 0.55, 0.25, 0.08],
                default=0.0,
            )
        )
    )


def objective_score(
    *,
    simulated_hit_rate: float,
    average_pair_similarity: float,
    unique_ticket_ratio: float,
    average_value_index: float,
    tail_score: float,
    config: MonteCarloConfig,
) -> float:
    """Calculate the portfolio optimization objective."""
    diversity_score = (
        0.5 * (1.0 - average_pair_similarity)
        + 0.5 * unique_ticket_ratio
    )
    return float(
        config.hit_rate_weight * simulated_hit_rate
        + config.diversity_weight * diversity_score
        + config.value_weight
        * max(average_value_index - 1.0, 0.0)
        + config.tail_weight * tail_score
    )


def evaluate_portfolio(
    portfolio: pd.DataFrame,
    simulations: np.ndarray,
    config: MonteCarloConfig,
) -> PortfolioMetrics:
    """Evaluate one portfolio against fixed simulation paths."""
    encoded = encode_tickets(portfolio["picks"].tolist())
    similarity = pairwise_similarity_matrix(encoded)

    upper_triangle = similarity[
        np.triu_indices(len(encoded), k=1)
    ]
    average_similarity = (
        float(upper_triangle.mean())
        if upper_triangle.size
        else 0.0
    )
    maximum_similarity = (
        float(upper_triangle.max())
        if upper_triangle.size
        else 0.0
    )

    best_counts = portfolio_best_match_counts(
        encoded,
        simulations,
    )
    simulated_hit_rate = float(np.mean(best_counts == 13))
    probability_10_or_more = float(np.mean(best_counts >= 10))
    probability_11_or_more = float(np.mean(best_counts >= 11))
    probability_12_or_more = float(np.mean(best_counts >= 12))
    probability_13 = simulated_hit_rate
    unique_ratio = float(
        portfolio["picks"].nunique() / len(portfolio)
    )
    average_value = float(
        portfolio["conservative_value_index"].mean()
    )
    tail_score = tail_coverage_score(best_counts)

    score = objective_score(
        simulated_hit_rate=simulated_hit_rate,
        average_pair_similarity=average_similarity,
        unique_ticket_ratio=unique_ratio,
        average_value_index=average_value,
        tail_score=tail_score,
        config=config,
    )

    if unique_ratio < config.minimum_unique_ratio:
        score -= (
            config.minimum_unique_ratio - unique_ratio
        )
    if maximum_similarity > config.maximum_pair_similarity:
        score -= (
            maximum_similarity
            - config.maximum_pair_similarity
        )

    return PortfolioMetrics(
        ticket_count=len(portfolio),
        exact_hit_probability=exact_portfolio_probability(
            portfolio
        ),
        simulated_hit_rate=simulated_hit_rate,
        mean_best_match_count=float(best_counts.mean()),
        probability_10_or_more=probability_10_or_more,
        probability_11_or_more=probability_11_or_more,
        probability_12_or_more=probability_12_or_more,
        probability_13=probability_13,
        average_pair_similarity=average_similarity,
        maximum_pair_similarity=maximum_similarity,
        unique_ticket_ratio=unique_ratio,
        average_value_index=average_value,
        tail_coverage_score=tail_score,
        objective_score=score,
    )


def initial_portfolio(
    candidates: pd.DataFrame,
    ticket_count: int,
) -> pd.DataFrame:
    """Build the initial portfolio from top-ranked candidates."""
    return candidates.head(ticket_count).copy().reset_index(drop=True)


def quick_surrogate_score(
    portfolio: pd.DataFrame,
    config: MonteCarloConfig,
) -> float:
    """
    Fast score used inside replacement search.

    Full Monte Carlo evaluation is intentionally avoided on every iteration.
    """
    encoded = encode_tickets(portfolio["picks"].tolist())
    similarity = pairwise_similarity_matrix(encoded)
    upper = similarity[np.triu_indices(len(encoded), k=1)]

    average_similarity = (
        float(upper.mean()) if upper.size else 0.0
    )
    maximum_similarity = (
        float(upper.max()) if upper.size else 0.0
    )
    unique_ratio = float(
        portfolio["picks"].nunique() / len(portfolio)
    )
    model_coverage = float(
        portfolio["model_probability"].sum()
    )
    average_value = float(
        portfolio["conservative_value_index"].mean()
    )

    score = (
        0.55 * model_coverage
        + 0.20 * (1.0 - average_similarity)
        + 0.20 * max(average_value - 1.0, 0.0)
        + 0.05 * unique_ratio
    )

    if maximum_similarity > config.maximum_pair_similarity:
        score -= (
            maximum_similarity
            - config.maximum_pair_similarity
        )

    return float(score)


def optimize_portfolio(
    candidates: pd.DataFrame,
    simulations: np.ndarray,
    config: MonteCarloConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, PortfolioMetrics, PortfolioMetrics]:
    """
    Optimize a ticket portfolio using stochastic replacement search.

    The search uses a fast surrogate on every iteration, then validates the
    final portfolio with the full Monte Carlo simulation.
    """
    pool = candidates.head(
        min(config.replacement_pool_size, len(candidates))
    ).copy().reset_index(drop=True)

    current = initial_portfolio(
        pool,
        config.ticket_count,
    )
    initial_metrics = evaluate_portfolio(
        current,
        simulations,
        config,
    )

    current_score = quick_surrogate_score(current, config)
    best = current.copy()
    best_score = current_score

    selected_picks = set(current["picks"])
    temperature_start = 0.02

    for iteration in range(config.optimization_iterations):
        remove_index = int(
            rng.integers(0, len(current))
        )
        candidate_index = int(
            rng.integers(0, len(pool))
        )
        candidate_row = pool.iloc[candidate_index]

        if candidate_row["picks"] in selected_picks:
            continue

        proposal = current.copy()
        removed_pick = str(
            proposal.iloc[remove_index]["picks"]
        )
        proposal.iloc[remove_index] = candidate_row

        proposal_score = quick_surrogate_score(
            proposal,
            config,
        )
        delta = proposal_score - current_score

        progress = iteration / max(
            config.optimization_iterations - 1,
            1,
        )
        temperature = (
            temperature_start * (1.0 - progress)
            + 1e-6
        )

        accept = (
            delta >= 0.0
            or rng.random()
            < math.exp(delta / temperature)
        )

        if accept:
            current = proposal
            current_score = proposal_score
            selected_picks.remove(removed_pick)
            selected_picks.add(
                str(candidate_row["picks"])
            )

            if current_score > best_score:
                best = current.copy()
                best_score = current_score

        if (
            iteration > 0
            and iteration % 1_000 == 0
        ):
            LOGGER.info(
                "Optimization iteration %d/%d, best surrogate=%.8f",
                iteration,
                config.optimization_iterations,
                best_score,
            )

    optimized_metrics = evaluate_portfolio(
        best,
        simulations,
        config,
    )

    return (
        best.sort_values(
            ["search_score", "model_probability"],
            ascending=[False, False],
            kind="stable",
        ).reset_index(drop=True),
        initial_metrics,
        optimized_metrics,
    )


def build_ticket_output(
    portfolio: pd.DataFrame,
    *,
    round_id: int,
    ticket_price_yen: int,
) -> pd.DataFrame:
    """Build final ticket output with one column per toto match."""
    rows: list[dict[str, object]] = []

    for ticket_number, row in enumerate(
        portfolio.to_dict(orient="records"),
        start=1,
    ):
        ticket = str(row["picks"])
        output: dict[str, object] = {
            "round_id": round_id,
            "ticket_number": ticket_number,
            "ticket_cost_yen": ticket_price_yen,
            "picks": ticket,
            "model_probability": row[
                "model_probability"
            ],
            "market_probability": row[
                "market_probability"
            ],
            "conservative_value_index": row[
                "conservative_value_index"
            ],
            "search_score": row["search_score"],
        }

        for match_number, pick in enumerate(
            ticket,
            start=1,
        ):
            output[f"match_{match_number:02d}"] = pick

        rows.append(output)

    return pd.DataFrame(rows)


def save_outputs(
    portfolio: pd.DataFrame,
    summary: OptimizationSummary,
    config: MonteCarloConfig,
) -> None:
    """Save final tickets and optimization summaries."""
    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ticket_output = build_ticket_output(
        portfolio,
        round_id=summary.round_id,
        ticket_price_yen=config.ticket_price_yen,
    )
    ticket_output.to_csv(
        config.output_dir
        / "current_round_tickets_monte_carlo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_dict = asdict(summary)

    (
        config.output_dir
        / "monte_carlo_optimizer_summary.json"
    ).write_text(
        json.dumps(
            summary_dict,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.json_normalize(summary_dict).to_csv(
        config.output_dir
        / "monte_carlo_optimizer_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    root = project_root()
    default_dir = (
        root
        / "ml"
        / "current_round"
        / "optimizer_v6"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo portfolio optimizer for Project Alpha."
        )
    )
    parser.add_argument(
        "--confidence-csv",
        type=Path,
        default=(
            default_dir
            / "current_round_confidence_v6.csv"
        ),
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=(
            default_dir
            / "current_round_candidate_ranking_v6.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_dir,
    )
    parser.add_argument(
        "--ticket-count",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    """Run Monte Carlo ticket portfolio optimization."""
    args = parse_args()
    configure_logging(args.verbose)

    config = MonteCarloConfig(
        confidence_csv=args.confidence_csv,
        candidate_csv=args.candidate_csv,
        output_dir=args.output_dir,
        ticket_count=args.ticket_count,
        simulations=args.simulations,
        optimization_iterations=args.iterations,
        replacement_pool_size=args.pool_size,
        seed=args.seed,
    )
    config.validate()

    confidence = load_confidence_data(
        config.confidence_csv
    )
    candidates = load_candidates(
        config.candidate_csv,
        minimum_count=config.ticket_count,
    )

    round_id = int(
        confidence["round_id"].iloc[0]
    )
    rng = np.random.default_rng(config.seed)

    probabilities = probability_tensor(
        confidence,
        config,
    )
    simulations = simulate_outcomes(
        probabilities,
        simulations=config.simulations,
        rng=rng,
    )

    optimized, initial_metrics, optimized_metrics = (
        optimize_portfolio(
            candidates=candidates,
            simulations=simulations,
            config=config,
            rng=rng,
        )
    )

    summary = OptimizationSummary(
        round_id=round_id,
        seed=config.seed,
        simulations=config.simulations,
        optimization_iterations=(
            config.optimization_iterations
        ),
        ticket_count=len(optimized),
        investment_yen=(
            len(optimized) * config.ticket_price_yen
        ),
        initial_metrics=initial_metrics,
        optimized_metrics=optimized_metrics,
        objective_improvement=(
            optimized_metrics.objective_score
            - initial_metrics.objective_score
        ),
        hit_rate_improvement=(
            optimized_metrics.simulated_hit_rate
            - initial_metrics.simulated_hit_rate
        ),
        probability_11_plus_improvement=(
            optimized_metrics.probability_11_or_more
            - initial_metrics.probability_11_or_more
        ),
        probability_12_plus_improvement=(
            optimized_metrics.probability_12_or_more
            - initial_metrics.probability_12_or_more
        ),
    )

    save_outputs(
        portfolio=optimized,
        summary=summary,
        config=config,
    )

    print("=" * 92)
    print("Project Alpha Monte Carlo Portfolio Optimizer")
    print("=" * 92)
    print(f"Round ID                  : {summary.round_id}")
    print(f"Simulations               : {summary.simulations:,}")
    print(
        "Optimization iterations  : "
        f"{summary.optimization_iterations:,}"
    )
    print(f"Ticket count              : {summary.ticket_count}")
    print(
        "Investment                : "
        f"{summary.investment_yen:,} yen"
    )
    print(
        "Initial objective         : "
        f"{initial_metrics.objective_score:.8f}"
    )
    print(
        "Optimized objective       : "
        f"{optimized_metrics.objective_score:.8f}"
    )
    print(
        "Objective improvement     : "
        f"{summary.objective_improvement:+.8f}"
    )
    print(
        "Initial P(11+)            : "
        f"{initial_metrics.probability_11_or_more:.6%}"
    )
    print(
        "Optimized P(11+)          : "
        f"{optimized_metrics.probability_11_or_more:.6%}"
    )
    print(
        "Initial P(12+)            : "
        f"{initial_metrics.probability_12_or_more:.6%}"
    )
    print(
        "Optimized P(12+)          : "
        f"{optimized_metrics.probability_12_or_more:.6%}"
    )
    print(
        "Initial exact hit rate    : "
        f"{initial_metrics.simulated_hit_rate:.6%}"
    )
    print(
        "Optimized exact hit rate  : "
        f"{optimized_metrics.simulated_hit_rate:.6%}"
    )
    print(
        "Average pair similarity  : "
        f"{optimized_metrics.average_pair_similarity:.4f}"
    )
    print(f"Outputs                   : {config.output_dir}")


if __name__ == "__main__":
    main()
