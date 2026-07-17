from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("current_round_confidence_optimizer_v6")

OUTCOMES = ("A", "D", "H")
AI_COLUMNS = {
    "A": "prob_away",
    "D": "prob_draw",
    "H": "prob_home",
}
MARKET_COLUMNS = {
    "A": "market_prob_away",
    "D": "market_prob_draw",
    "H": "market_prob_home",
}


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Confidence and ROI optimizer configuration."""

    input_csv: Path
    output_dir: Path
    ticket_price_yen: int = 100
    normal_budget_yen: int = 5_000
    maximum_budget_yen: int = 10_000
    rollover_balance_yen: int = 0
    rollover_cap_yen: int = 5_000
    minimum_portfolio_value_index: float = 1.10
    probability_shrinkage: float = 0.35
    per_match_value_cap: float = 1.50
    beam_width: int = 30_000
    candidate_limit: int = 8_000
    diversity_penalty: float = 0.10
    value_weight: float = 0.40
    confidence_weight: float = 0.20
    disagreement_penalty: float = 0.12

    def validate(self) -> None:
        if not self.input_csv.exists():
            raise FileNotFoundError(
                f"Input CSV not found: {self.input_csv}"
            )
        if self.ticket_price_yen <= 0:
            raise ValueError("ticket_price_yen must be positive.")
        if self.normal_budget_yen <= 0:
            raise ValueError("normal_budget_yen must be positive.")
        if self.maximum_budget_yen < self.normal_budget_yen:
            raise ValueError(
                "maximum_budget_yen must be >= normal_budget_yen."
            )
        if self.rollover_balance_yen < 0:
            raise ValueError("rollover_balance_yen must not be negative.")
        if self.rollover_cap_yen < 0:
            raise ValueError("rollover_cap_yen must not be negative.")
        if self.minimum_portfolio_value_index <= 0:
            raise ValueError(
                "minimum_portfolio_value_index must be positive."
            )
        if not 0.0 <= self.probability_shrinkage <= 1.0:
            raise ValueError(
                "probability_shrinkage must be between 0 and 1."
            )
        if self.per_match_value_cap < 1.0:
            raise ValueError(
                "per_match_value_cap must be at least 1.0."
            )


@dataclass(frozen=True, slots=True)
class TicketCandidate:
    """One exact 13-match ticket candidate."""

    picks: str
    model_probability: float
    market_probability: float
    conservative_value_index: float
    search_score: float


@dataclass(frozen=True, slots=True)
class PortfolioSummary:
    """Selected portfolio summary."""

    round_id: int
    decision: str
    reason: str
    investment_yen: int
    ticket_count: int
    available_budget_yen: int
    normal_budget_yen: int
    rollover_before_yen: int
    rollover_after_yen: int
    estimated_portfolio_value_index: float | None
    estimated_expected_return_yen: float | None
    estimated_expected_profit_yen: float | None
    model_coverage_probability: float
    market_coverage_probability: float
    high_confidence_matches: int
    medium_confidence_matches: int
    low_confidence_matches: int


@dataclass(slots=True)
class BeamState:
    """Partial ticket state used in beam search."""

    picks: str
    log_model_probability: float
    log_market_probability: float
    search_score: float


class ConfidenceOptimizerError(RuntimeError):
    """Raised when current-round optimization cannot complete safely."""


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    required = {
        "round_id",
        "toto_match_no",
        "match_card_id",
        "home_team",
        "away_team",
        *AI_COLUMNS.values(),
        *MARKET_COLUMNS.values(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ConfidenceOptimizerError(
            "Input CSV missing required columns: "
            + ", ".join(missing)
        )

    if len(frame) != 13:
        raise ConfidenceOptimizerError(
            f"Expected 13 rows, found {len(frame)}."
        )

    frame = frame.sort_values(
        "toto_match_no",
        kind="stable",
    ).reset_index(drop=True)

    expected = list(range(1, 14))
    actual = frame["toto_match_no"].astype(int).tolist()
    if actual != expected:
        raise ConfidenceOptimizerError(
            f"Expected toto_match_no 1..13, found {actual}."
        )

    if frame["match_card_id"].duplicated().any():
        raise ConfidenceOptimizerError(
            "Duplicate match_card_id values found."
        )

    for columns, label in (
        (list(AI_COLUMNS.values()), "AI"),
        (list(MARKET_COLUMNS.values()), "Market"),
    ):
        values = frame[columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if values.isna().any().any():
            raise ConfidenceOptimizerError(
                f"{label} probabilities contain missing values."
            )
        if (values < 0).any().any():
            raise ConfidenceOptimizerError(
                f"{label} probabilities contain negative values."
            )
        totals = values.sum(axis=1)
        if (totals <= 0).any():
            raise ConfidenceOptimizerError(
                f"{label} probabilities contain zero-sum rows."
            )
        frame[columns] = values.div(totals, axis=0)

    return frame


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return entropy / math.log(probabilities.shape[1])


def jensen_shannon_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    midpoint = 0.5 * (first + second)
    first = np.clip(first, 1e-15, 1.0)
    second = np.clip(second, 1e-15, 1.0)
    midpoint = np.clip(midpoint, 1e-15, 1.0)

    kl_first = np.sum(
        first * np.log(first / midpoint),
        axis=1,
    )
    kl_second = np.sum(
        second * np.log(second / midpoint),
        axis=1,
    )
    divergence = 0.5 * (kl_first + kl_second)
    return np.sqrt(
        np.clip(divergence / math.log(2.0), 0.0, 1.0)
    )


def build_confidence_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()

    ai = output[list(AI_COLUMNS.values())].to_numpy(dtype=float)
    sorted_ai = np.sort(ai, axis=1)[:, ::-1]
    order = np.argsort(-ai, axis=1)

    output["primary_pick"] = [
        OUTCOMES[index] for index in order[:, 0]
    ]
    output["secondary_pick"] = [
        OUTCOMES[index] for index in order[:, 1]
    ]
    output["top_probability"] = sorted_ai[:, 0]
    output["second_probability"] = sorted_ai[:, 1]
    output["probability_margin"] = (
        sorted_ai[:, 0] - sorted_ai[:, 1]
    )
    output["probability_entropy"] = normalized_entropy(ai)
    output["certainty_from_entropy"] = (
        1.0 - output["probability_entropy"]
    )

    if all(
        column in output.columns
        for column in (
            "rf_prob_away",
            "rf_prob_draw",
            "rf_prob_home",
            "lgbm_prob_away",
            "lgbm_prob_draw",
            "lgbm_prob_home",
        )
    ):
        rf = output[
            ["rf_prob_away", "rf_prob_draw", "rf_prob_home"]
        ].to_numpy(dtype=float)
        lgbm = output[
            ["lgbm_prob_away", "lgbm_prob_draw", "lgbm_prob_home"]
        ].to_numpy(dtype=float)
        output["model_disagreement_js"] = jensen_shannon_distance(
            rf,
            lgbm,
        )
        output["models_agree_top_pick"] = (
            rf.argmax(axis=1) == lgbm.argmax(axis=1)
        )
    else:
        output["model_disagreement_js"] = 0.0
        output["models_agree_top_pick"] = True

    margin_component = np.clip(
        output["probability_margin"] / 0.35,
        0.0,
        1.0,
    )
    top_component = np.clip(
        (output["top_probability"] - (1.0 / 3.0))
        / (2.0 / 3.0),
        0.0,
        1.0,
    )
    agreement_component = (
        1.0 - output["model_disagreement_js"]
    )
    value_component = np.clip(
        output.get("best_value_ratio", 1.0) - 1.0,
        0.0,
        1.0,
    )

    output["confidence_score"] = (
        0.30 * output["certainty_from_entropy"]
        + 0.25 * margin_component
        + 0.20 * agreement_component
        + 0.15 * top_component
        + 0.10 * value_component
    ).clip(0.0, 1.0)

    output["confidence_level"] = np.select(
        [
            output["confidence_score"] >= 0.45,
            output["confidence_score"] >= 0.30,
        ],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )

    single = (
        (output["top_probability"] >= 0.48)
        & (output["probability_margin"] >= 0.08)
        & (output["model_disagreement_js"] <= 0.14)
    )
    double = (
        (output["top_probability"] >= 0.40)
        & (output["probability_margin"] >= 0.03)
        & (output["model_disagreement_js"] <= 0.20)
    )
    output["coverage_recommendation"] = np.select(
        [single, double],
        ["SINGLE", "DOUBLE"],
        default="TRIPLE",
    )

    combinations: list[str] = []
    for row_index, coverage in enumerate(
        output["coverage_recommendation"]
    ):
        if coverage == "SINGLE":
            combinations.append(
                OUTCOMES[order[row_index, 0]]
            )
        elif coverage == "DOUBLE":
            combinations.append(
                "".join(
                    OUTCOMES[index]
                    for index in order[row_index, :2]
                )
            )
        else:
            combinations.append("ADH")

    output["recommended_combination"] = combinations
    output["uncertainty_score"] = (
        1.0 - output["confidence_score"]
    )

    if "best_edge" not in output.columns:
        edge_values = np.column_stack(
            [
                output["prob_away"]
                - output["market_prob_away"],
                output["prob_draw"]
                - output["market_prob_draw"],
                output["prob_home"]
                - output["market_prob_home"],
            ]
        )
        output["best_edge"] = edge_values.max(axis=1)

    if "best_value_ratio" not in output.columns:
        value_values = np.column_stack(
            [
                output["prob_away"]
                / output["market_prob_away"].clip(lower=1e-15),
                output["prob_draw"]
                / output["market_prob_draw"].clip(lower=1e-15),
                output["prob_home"]
                / output["market_prob_home"].clip(lower=1e-15),
            ]
        )
        output["best_value_ratio"] = value_values.max(axis=1)

    output["roi_priority_score"] = (
        output["confidence_score"]
        * output["best_edge"].clip(lower=0.0)
        * output["best_value_ratio"].clip(lower=0.0, upper=3.0)
    )

    return output


def available_budget(config: OptimizerConfig) -> int:
    usable_rollover = min(
        config.rollover_balance_yen,
        config.rollover_cap_yen,
    )
    return min(
        config.maximum_budget_yen,
        config.normal_budget_yen + usable_rollover,
    )


def option_score(
    model_probability: float,
    market_probability: float,
    confidence: float,
    disagreement: float,
    config: OptimizerConfig,
) -> float:
    value_ratio = model_probability / max(
        market_probability,
        1e-15,
    )
    return (
        math.log(max(model_probability, 1e-15))
        + config.value_weight
        * math.log(max(value_ratio, 1e-15))
        + config.confidence_weight * confidence
        - config.disagreement_penalty * disagreement
    )


def generate_candidates(
    frame: pd.DataFrame,
    config: OptimizerConfig,
) -> list[TicketCandidate]:
    beam = [
        BeamState(
            picks="",
            log_model_probability=0.0,
            log_market_probability=0.0,
            search_score=0.0,
        )
    ]

    for _, row in frame.iterrows():
        expanded: list[BeamState] = []

        for state in beam:
            for outcome in OUTCOMES:
                model_probability = float(
                    row[AI_COLUMNS[outcome]]
                )
                market_probability = float(
                    row[MARKET_COLUMNS[outcome]]
                )
                increment = option_score(
                    model_probability=model_probability,
                    market_probability=market_probability,
                    confidence=float(row["confidence_score"]),
                    disagreement=float(
                        row["model_disagreement_js"]
                    ),
                    config=config,
                )
                expanded.append(
                    BeamState(
                        picks=state.picks + outcome,
                        log_model_probability=(
                            state.log_model_probability
                            + math.log(
                                max(model_probability, 1e-15)
                            )
                        ),
                        log_market_probability=(
                            state.log_market_probability
                            + math.log(
                                max(market_probability, 1e-15)
                            )
                        ),
                        search_score=(
                            state.search_score + increment
                        ),
                    )
                )

        beam = heapq.nlargest(
            config.beam_width,
            expanded,
            key=lambda state: state.search_score,
        )

    candidates: list[TicketCandidate] = []
    for state in heapq.nlargest(
        config.candidate_limit,
        beam,
        key=lambda item: item.search_score,
    ):
        model_probability = math.exp(
            state.log_model_probability
        )
        market_probability = math.exp(
            state.log_market_probability
        )
        leg_value_ratios: list[float] = []
        for match_index, outcome in enumerate(state.picks):
            row = frame.iloc[match_index]
            model_probability_leg = float(
                row[AI_COLUMNS[outcome]]
            )
            market_probability_leg = float(
                row[MARKET_COLUMNS[outcome]]
            )
            conservative_probability_leg = (
                market_probability_leg
                + config.probability_shrinkage
                * (
                    model_probability_leg
                    - market_probability_leg
                )
            )
            ratio = (
                conservative_probability_leg
                / max(market_probability_leg, 1e-15)
            )
            leg_value_ratios.append(
                min(max(ratio, 1e-12), config.per_match_value_cap)
            )

        conservative_value_index = float(
            math.exp(
                sum(math.log(value) for value in leg_value_ratios)
                / len(leg_value_ratios)
            )
        )

        candidates.append(
            TicketCandidate(
                picks=state.picks,
                model_probability=model_probability,
                market_probability=market_probability,
                conservative_value_index=conservative_value_index,
                search_score=state.search_score,
            )
        )

    return candidates


def ticket_similarity(first: str, second: str) -> float:
    return (
        sum(
            left == right
            for left, right in zip(
                first,
                second,
                strict=True,
            )
        )
        / len(first)
    )


def select_diverse_tickets(
    candidates: list[TicketCandidate],
    ticket_count: int,
    config: OptimizerConfig,
) -> list[TicketCandidate]:
    selected: list[TicketCandidate] = []
    remaining = candidates.copy()

    while len(selected) < ticket_count:
        best_index = -1
        best_score = -math.inf

        for index, candidate in enumerate(remaining):
            max_similarity = (
                max(
                    ticket_similarity(
                        candidate.picks,
                        chosen.picks,
                    )
                    for chosen in selected
                )
                if selected
                else 0.0
            )
            adjusted = (
                candidate.search_score
                - config.diversity_penalty * max_similarity
            )
            if adjusted > best_score:
                best_score = adjusted
                best_index = index

        if best_index < 0:
            break
        selected.append(remaining.pop(best_index))

    return selected


def budget_levels(
    available_yen: int,
    ticket_price_yen: int,
) -> list[int]:
    standard = (
        10_000,
        7_500,
        5_000,
        3_000,
        2_000,
        1_000,
    )
    levels = {
        amount
        for amount in standard
        if amount <= available_yen
        and amount % ticket_price_yen == 0
    }
    rounded_available = (
        available_yen
        - available_yen % ticket_price_yen
    )
    if rounded_available > 0:
        levels.add(rounded_available)

    return sorted(levels, reverse=True)


def portfolio_value_index(
    tickets: list[TicketCandidate],
) -> float | None:
    if not tickets:
        return None
    return float(
        np.mean(
            [
                ticket.conservative_value_index
                for ticket in tickets
            ]
        )
    )


def choose_portfolio(
    candidates: list[TicketCandidate],
    config: OptimizerConfig,
) -> tuple[int, list[TicketCandidate], str]:
    budget = available_budget(config)

    for investment in budget_levels(
        budget,
        config.ticket_price_yen,
    ):
        ticket_count = (
            investment // config.ticket_price_yen
        )
        selected = select_diverse_tickets(
            candidates=candidates,
            ticket_count=ticket_count,
            config=config,
        )
        estimated_value = portfolio_value_index(selected)

        if (
            estimated_value is not None
            and estimated_value
            >= config.minimum_portfolio_value_index
        ):
            return (
                investment,
                selected,
                (
                    f"Conservative portfolio value index "
                    f"{estimated_value:.4f} >= "
                    f"{config.minimum_portfolio_value_index:.4f}."
                ),
            )

    return (
        0,
        [],
        (
            "No candidate budget maintained the minimum "
            f"portfolio EV of "
            f"{config.minimum_portfolio_value_index:.4f}."
        ),
    )


def rollover_after(
    investment_yen: int,
    config: OptimizerConfig,
) -> int:
    before = min(
        config.rollover_balance_yen,
        config.rollover_cap_yen,
    )

    if investment_yen <= config.normal_budget_yen:
        unused_normal = (
            config.normal_budget_yen
            - investment_yen
        )
        return min(
            config.rollover_cap_yen,
            before + unused_normal,
        )

    used_rollover = (
        investment_yen
        - config.normal_budget_yen
    )
    return max(0, before - used_rollover)


def build_ticket_frame(
    tickets: list[TicketCandidate],
    round_id: int,
    ticket_price_yen: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for number, ticket in enumerate(tickets, start=1):
        row: dict[str, Any] = {
            "round_id": round_id,
            "ticket_number": number,
            "ticket_cost_yen": ticket_price_yen,
            "picks": ticket.picks,
            "model_probability": (
                ticket.model_probability
            ),
            "market_probability": (
                ticket.market_probability
            ),
            "conservative_value_index": (
                ticket.conservative_value_index
            ),
            "search_score": ticket.search_score,
        }
        for index, pick in enumerate(
            ticket.picks,
            start=1,
        ):
            row[f"match_{index:02d}"] = pick
        rows.append(row)

    return pd.DataFrame(rows)


def build_summary(
    frame: pd.DataFrame,
    investment: int,
    tickets: list[TicketCandidate],
    reason: str,
    config: OptimizerConfig,
) -> PortfolioSummary:
    estimated_value = portfolio_value_index(tickets)
    expected_return = None
    expected_profit = None

    return PortfolioSummary(
        round_id=int(frame["round_id"].iloc[0]),
        decision="BUY" if investment > 0 else "SKIP",
        reason=reason,
        investment_yen=investment,
        ticket_count=len(tickets),
        available_budget_yen=available_budget(config),
        normal_budget_yen=config.normal_budget_yen,
        rollover_before_yen=(
            config.rollover_balance_yen
        ),
        rollover_after_yen=rollover_after(
            investment,
            config,
        ),
        estimated_portfolio_value_index=estimated_value,
        estimated_expected_return_yen=expected_return,
        estimated_expected_profit_yen=expected_profit,
        model_coverage_probability=float(
            sum(
                ticket.model_probability
                for ticket in tickets
            )
        ),
        market_coverage_probability=float(
            sum(
                ticket.market_probability
                for ticket in tickets
            )
        ),
        high_confidence_matches=int(
            (frame["confidence_level"] == "HIGH").sum()
        ),
        medium_confidence_matches=int(
            (frame["confidence_level"] == "MEDIUM").sum()
        ),
        low_confidence_matches=int(
            (frame["confidence_level"] == "LOW").sum()
        ),
    )


def save_outputs(
    frame: pd.DataFrame,
    candidates: list[TicketCandidate],
    tickets: list[TicketCandidate],
    summary: PortfolioSummary,
    config: OptimizerConfig,
) -> None:
    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        config.output_dir
        / "current_round_confidence_v6.csv",
        index=False,
        encoding="utf-8-sig",
    )

    build_ticket_frame(
        tickets=tickets,
        round_id=summary.round_id,
        ticket_price_yen=config.ticket_price_yen,
    ).to_csv(
        config.output_dir
        / "current_round_tickets_v6.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "rank": rank,
                **asdict(candidate),
            }
            for rank, candidate in enumerate(
                candidates[:500],
                start=1,
            )
        ]
    ).to_csv(
        config.output_dir
        / "current_round_candidate_ranking_v6.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([asdict(summary)]).to_csv(
        config.output_dir
        / "current_round_optimizer_v6_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        config.output_dir
        / "current_round_optimizer_v6_summary.json"
    ).write_text(
        json.dumps(
            asdict(summary),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        config.output_dir
        / "rollover_state_next.json"
    ).write_text(
        json.dumps(
            {
                "round_id": summary.round_id,
                "rollover_balance_yen": (
                    summary.rollover_after_yen
                ),
                "last_decision": summary.decision,
                "last_investment_yen": (
                    summary.investment_yen
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(
        description=(
            "Connect current-round Confidence Engine and "
            "ROI Optimizer for Project Alpha v6."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            root
            / "ml"
            / "current_round"
            / "current_round_market_predictions_v6.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            root
            / "ml"
            / "current_round"
            / "optimizer_v6"
        ),
    )
    parser.add_argument(
        "--rollover-balance",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--minimum-ev",
        type=float,
        default=1.10,
    )
    parser.add_argument(
        "--normal-budget",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--maximum-budget",
        type=int,
        default=10_000,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    config = OptimizerConfig(
        input_csv=args.input,
        output_dir=args.output_dir,
        normal_budget_yen=args.normal_budget,
        maximum_budget_yen=args.maximum_budget,
        rollover_balance_yen=args.rollover_balance,
        minimum_portfolio_value_index=args.minimum_ev,
    )
    config.validate()

    frame = load_input(config.input_csv)
    frame = build_confidence_features(frame)

    candidates = generate_candidates(
        frame=frame,
        config=config,
    )
    investment, tickets, reason = choose_portfolio(
        candidates=candidates,
        config=config,
    )
    summary = build_summary(
        frame=frame,
        investment=investment,
        tickets=tickets,
        reason=reason,
        config=config,
    )

    save_outputs(
        frame=frame,
        candidates=candidates,
        tickets=tickets,
        summary=summary,
        config=config,
    )

    print("=" * 92)
    print("Project Alpha Confidence + ROI Optimizer v6")
    print("=" * 92)
    print(f"Round ID                  : {summary.round_id}")
    print(f"Decision                  : {summary.decision}")
    print(f"Reason                    : {summary.reason}")
    print(
        f"Available budget          : "
        f"{summary.available_budget_yen:,} yen"
    )
    print(
        f"Investment                : "
        f"{summary.investment_yen:,} yen"
    )
    print(f"Ticket count              : {summary.ticket_count}")
    print(
        f"Conservative portfolio value index    : "
        f"{summary.estimated_portfolio_value_index}"
    )
    print(
        f"Expected return           : "
        f"{summary.estimated_expected_return_yen}"
    )
    print(
        f"Expected profit           : "
        f"{summary.estimated_expected_profit_yen}"
    )
    print(
        f"High/Medium/Low matches   : "
        f"{summary.high_confidence_matches}/"
        f"{summary.medium_confidence_matches}/"
        f"{summary.low_confidence_matches}"
    )
    print(
        f"Rollover before/after     : "
        f"{summary.rollover_before_yen:,}/"
        f"{summary.rollover_after_yen:,} yen"
    )
    print(f"Outputs                   : {config.output_dir}")


if __name__ == "__main__":
    main()
