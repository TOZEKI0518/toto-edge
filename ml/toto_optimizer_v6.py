from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("toto_optimizer_v6")

OUTCOMES = ("A", "D", "H")
MODEL_COLUMNS = {
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
    """Configuration for one 13-match toto optimization run."""

    input_csv: Path
    output_dir: Path
    market_csv: Path | None = None
    round_id: str = "current"
    ticket_price_yen: int = 100
    normal_budget_yen: int = 5_000
    maximum_budget_yen: int = 10_000
    rollover_cap_yen: int = 5_000
    rollover_balance_yen: int = 0
    minimum_portfolio_ev: float = 1.10
    payout_rate: float = 0.50
    beam_width: int = 20_000
    candidate_limit: int = 5_000
    value_weight: float = 0.35
    confidence_weight: float = 0.12
    disagreement_penalty: float = 0.10
    diversity_penalty: float = 0.08
    allow_without_market: bool = False

    def validate(self) -> None:
        if not self.input_csv.exists():
            raise FileNotFoundError(
                f"Input prediction CSV was not found: {self.input_csv}"
            )
        if self.market_csv is not None and not self.market_csv.exists():
            raise FileNotFoundError(
                f"Market CSV was not found: {self.market_csv}"
            )
        if self.ticket_price_yen <= 0:
            raise ValueError("ticket_price_yen must be positive.")
        if self.normal_budget_yen <= 0:
            raise ValueError("normal_budget_yen must be positive.")
        if self.maximum_budget_yen < self.normal_budget_yen:
            raise ValueError(
                "maximum_budget_yen must be at least normal_budget_yen."
            )
        if self.rollover_cap_yen < 0 or self.rollover_balance_yen < 0:
            raise ValueError("Rollover values must not be negative.")
        if self.minimum_portfolio_ev <= 0:
            raise ValueError("minimum_portfolio_ev must be positive.")
        if not 0 < self.payout_rate <= 1:
            raise ValueError("payout_rate must be in (0, 1].")
        if self.beam_width <= 0 or self.candidate_limit <= 0:
            raise ValueError("Beam and candidate limits must be positive.")
        for budget in (
            self.normal_budget_yen,
            self.maximum_budget_yen,
            self.rollover_cap_yen,
            self.rollover_balance_yen,
        ):
            if budget % self.ticket_price_yen:
                raise ValueError(
                    "All budget values must be multiples of ticket_price_yen."
                )


@dataclass(frozen=True, slots=True)
class TicketCandidate:
    """One exact 13-match outcome combination."""

    picks: str
    model_probability: float
    market_probability: float | None
    estimated_ev_ratio: float | None
    search_score: float


@dataclass(frozen=True, slots=True)
class PortfolioSummary:
    """Summary of the selected ticket portfolio."""

    round_id: str
    decision: str
    reason: str
    investment_yen: int
    ticket_count: int
    available_budget_yen: int
    normal_budget_yen: int
    rollover_before_yen: int
    rollover_after_yen: int
    estimated_portfolio_ev_ratio: float | None
    estimated_expected_return_yen: float | None
    estimated_expected_profit_yen: float | None
    model_coverage_probability: float
    market_coverage_probability: float | None
    mean_ticket_confidence: float
    mean_model_disagreement: float
    market_data_available: bool


@dataclass(slots=True)
class BeamState:
    """Partial ticket used during beam search."""

    picks: str
    log_model_probability: float
    log_market_probability: float
    search_score: float


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_probabilities(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(
            f"{label} probabilities contain missing/non-numeric values."
        )
    if (values < 0).any().any():
        raise ValueError(f"{label} probabilities contain negative values.")

    totals = values.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError(f"{label} probabilities contain zero-sum rows.")

    return values.div(totals, axis=0)


def load_round_predictions(config: OptimizerConfig) -> pd.DataFrame:
    """Load exactly 13 ordered matches for one toto round."""
    frame = pd.read_csv(config.input_csv)

    required = {
        "match_card_id",
        "home_team",
        "away_team",
        *MODEL_COLUMNS.values(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Input prediction CSV is missing required columns: "
            + ", ".join(missing)
        )

    if "round_id" in frame.columns:
        matching = frame.loc[
            frame["round_id"].astype(str) == str(config.round_id)
        ].copy()
        if not matching.empty:
            frame = matching

    if "toto_match_no" in frame.columns:
        frame = frame.sort_values("toto_match_no", kind="stable")
    elif "match_order" in frame.columns:
        frame = frame.sort_values("match_order", kind="stable")
    elif "match_date" in frame.columns:
        frame = frame.sort_values(
            ["match_date", "match_card_id"],
            kind="stable",
        )

    if len(frame) != 13:
        raise ValueError(
            "Optimizer input must contain exactly 13 matches. "
            f"Actual rows: {len(frame)}. Supply a round-specific CSV or "
            "a round_id column and --round-id."
        )

    if frame["match_card_id"].duplicated().any():
        raise ValueError("Input contains duplicate match_card_id values.")

    normalized = normalize_probabilities(
        frame,
        list(MODEL_COLUMNS.values()),
        label="Model",
    )
    frame = frame.reset_index(drop=True)
    frame[list(MODEL_COLUMNS.values())] = normalized.reset_index(drop=True)

    if "confidence_score" not in frame.columns:
        frame["confidence_score"] = frame[
            list(MODEL_COLUMNS.values())
        ].max(axis=1)
    frame["confidence_score"] = pd.to_numeric(
        frame["confidence_score"],
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)

    if "model_disagreement_js" not in frame.columns:
        frame["model_disagreement_js"] = 0.0
    frame["model_disagreement_js"] = pd.to_numeric(
        frame["model_disagreement_js"],
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)

    frame.insert(0, "toto_match_no", range(1, 14))
    return frame


def attach_market_probabilities(
    frame: pd.DataFrame,
    market_csv: Path | None,
) -> tuple[pd.DataFrame, bool]:
    """Attach public-pick probabilities when supplied."""
    has_embedded = all(
        column in frame.columns for column in MARKET_COLUMNS.values()
    )
    if has_embedded:
        normalized = normalize_probabilities(
            frame,
            list(MARKET_COLUMNS.values()),
            label="Market",
        )
        frame = frame.copy()
        frame[list(MARKET_COLUMNS.values())] = normalized
        return frame, True

    if market_csv is None:
        return frame, False

    market = pd.read_csv(market_csv)
    required = {"match_card_id", *MARKET_COLUMNS.values()}
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(
            "Market CSV is missing required columns: "
            + ", ".join(missing)
        )
    if market["match_card_id"].duplicated().any():
        raise ValueError("Market CSV contains duplicate match IDs.")

    normalized = normalize_probabilities(
        market,
        list(MARKET_COLUMNS.values()),
        label="Market",
    )
    market = market[["match_card_id"]].copy()
    market[list(MARKET_COLUMNS.values())] = normalized

    merged = frame.merge(
        market,
        on="match_card_id",
        how="left",
        validate="one_to_one",
    )
    missing_rows = int(
        merged[list(MARKET_COLUMNS.values())].isna().any(axis=1).sum()
    )
    if missing_rows:
        raise ValueError(
            f"Market probabilities are missing for {missing_rows} matches."
        )
    return merged, True


def option_search_score(
    model_probability: float,
    market_probability: float | None,
    confidence: float,
    disagreement: float,
    config: OptimizerConfig,
) -> float:
    """Score one outcome option during beam expansion."""
    score = math.log(max(model_probability, 1e-15))
    if market_probability is not None:
        value_ratio = model_probability / max(market_probability, 1e-15)
        score += config.value_weight * math.log(max(value_ratio, 1e-15))
    score += config.confidence_weight * confidence
    score -= config.disagreement_penalty * disagreement
    return score


def generate_candidates(
    frame: pd.DataFrame,
    market_available: bool,
    config: OptimizerConfig,
) -> list[TicketCandidate]:
    """Generate high-quality exact tickets using beam search."""
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
                model_probability = float(row[MODEL_COLUMNS[outcome]])
                market_probability = (
                    float(row[MARKET_COLUMNS[outcome]])
                    if market_available
                    else None
                )
                increment = option_search_score(
                    model_probability=model_probability,
                    market_probability=market_probability,
                    confidence=float(row["confidence_score"]),
                    disagreement=float(row["model_disagreement_js"]),
                    config=config,
                )
                expanded.append(
                    BeamState(
                        picks=state.picks + outcome,
                        log_model_probability=(
                            state.log_model_probability
                            + math.log(max(model_probability, 1e-15))
                        ),
                        log_market_probability=(
                            state.log_market_probability
                            + (
                                math.log(max(market_probability, 1e-15))
                                if market_probability is not None
                                else 0.0
                            )
                        ),
                        search_score=state.search_score + increment,
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
        model_probability = math.exp(state.log_model_probability)
        if market_available:
            market_probability = math.exp(state.log_market_probability)
            estimated_ev_ratio = (
                config.payout_rate
                * model_probability
                / max(market_probability, 1e-15)
            )
        else:
            market_probability = None
            estimated_ev_ratio = None

        candidates.append(
            TicketCandidate(
                picks=state.picks,
                model_probability=model_probability,
                market_probability=market_probability,
                estimated_ev_ratio=estimated_ev_ratio,
                search_score=state.search_score,
            )
        )

    return candidates


def ticket_similarity(first: str, second: str) -> float:
    """Return the proportion of identical selections across 13 matches."""
    if len(first) != len(second):
        raise ValueError("Ticket lengths must match.")
    return sum(a == b for a, b in zip(first, second, strict=True)) / len(first)


def select_diverse_tickets(
    candidates: list[TicketCandidate],
    ticket_count: int,
    config: OptimizerConfig,
) -> list[TicketCandidate]:
    """Greedily select valuable tickets while discouraging near duplicates."""
    if ticket_count <= 0:
        return []
    if len(candidates) < ticket_count:
        raise ValueError(
            f"Only {len(candidates)} candidates available for "
            f"{ticket_count} requested tickets."
        )

    selected: list[TicketCandidate] = []
    remaining = candidates.copy()

    while len(selected) < ticket_count:
        best_index = -1
        best_adjusted_score = -math.inf

        for index, candidate in enumerate(remaining):
            if selected:
                maximum_similarity = max(
                    ticket_similarity(candidate.picks, chosen.picks)
                    for chosen in selected
                )
            else:
                maximum_similarity = 0.0

            adjusted_score = (
                candidate.search_score
                - config.diversity_penalty * maximum_similarity
            )
            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_index = index

        selected.append(remaining.pop(best_index))

    return selected


def available_budget(config: OptimizerConfig) -> int:
    """Return current spendable amount including permitted rollover."""
    usable_rollover = min(
        config.rollover_balance_yen,
        config.rollover_cap_yen,
    )
    return min(
        config.maximum_budget_yen,
        config.normal_budget_yen + usable_rollover,
    )


def candidate_budget_levels(
    available_yen: int,
    config: OptimizerConfig,
) -> list[int]:
    """Return descending budget levels compatible with ticket price."""
    standard_levels = [
        10_000,
        7_500,
        5_000,
        3_000,
        2_000,
        1_000,
    ]
    levels = {
        level
        for level in standard_levels
        if level <= available_yen
        and level % config.ticket_price_yen == 0
    }
    levels.add(
        available_yen
        - (available_yen % config.ticket_price_yen)
    )
    return sorted(
        (level for level in levels if level > 0),
        reverse=True,
    )


def portfolio_ev_ratio(
    tickets: list[TicketCandidate],
) -> float | None:
    """Return equal-stake portfolio EV ratio."""
    values = [
        ticket.estimated_ev_ratio
        for ticket in tickets
        if ticket.estimated_ev_ratio is not None
    ]
    if len(values) != len(tickets) or not values:
        return None
    return float(np.mean(values))


def choose_investment(
    candidates: list[TicketCandidate],
    market_available: bool,
    config: OptimizerConfig,
) -> tuple[int, list[TicketCandidate], str]:
    """Select the largest budget whose estimated portfolio EV passes."""
    if not market_available and not config.allow_without_market:
        return (
            0,
            [],
            "Market probabilities unavailable; ROI-based purchase skipped.",
        )

    available_yen = available_budget(config)
    for budget_yen in candidate_budget_levels(available_yen, config):
        ticket_count = budget_yen // config.ticket_price_yen
        tickets = select_diverse_tickets(
            candidates=candidates,
            ticket_count=ticket_count,
            config=config,
        )

        if not market_available:
            return (
                min(budget_yen, config.normal_budget_yen),
                tickets,
                "Market-free exploratory purchase enabled.",
            )

        estimated_ev = portfolio_ev_ratio(tickets)
        if (
            estimated_ev is not None
            and estimated_ev >= config.minimum_portfolio_ev
        ):
            return (
                budget_yen,
                tickets,
                (
                    f"Estimated portfolio EV {estimated_ev:.4f} "
                    f">= threshold {config.minimum_portfolio_ev:.4f}."
                ),
            )

    return (
        0,
        [],
        (
            "No candidate budget maintained the minimum portfolio EV "
            f"of {config.minimum_portfolio_ev:.4f}."
        ),
    )


def rollover_after_purchase(
    investment_yen: int,
    config: OptimizerConfig,
) -> int:
    """Update rollover without exceeding the configured cap."""
    before = min(config.rollover_balance_yen, config.rollover_cap_yen)

    if investment_yen <= config.normal_budget_yen:
        unused_normal = config.normal_budget_yen - investment_yen
        return min(
            config.rollover_cap_yen,
            before + unused_normal,
        )

    rollover_used = investment_yen - config.normal_budget_yen
    return max(0, before - rollover_used)


def ticket_rows(
    selected: list[TicketCandidate],
    frame: pd.DataFrame,
    config: OptimizerConfig,
) -> pd.DataFrame:
    """Create user-facing ticket output rows."""
    rows: list[dict[str, Any]] = []
    for ticket_number, ticket in enumerate(selected, start=1):
        row: dict[str, Any] = {
            "round_id": config.round_id,
            "ticket_number": ticket_number,
            "ticket_cost_yen": config.ticket_price_yen,
            "picks": ticket.picks,
            "model_probability": ticket.model_probability,
            "market_probability": ticket.market_probability,
            "estimated_ev_ratio": ticket.estimated_ev_ratio,
            "search_score": ticket.search_score,
        }
        for match_index, pick in enumerate(ticket.picks, start=1):
            row[f"match_{match_index:02d}"] = pick
        rows.append(row)
    return pd.DataFrame(rows)


def match_analysis_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create one diagnostic row per match."""
    output = frame.copy()
    probability_values = output[list(MODEL_COLUMNS.values())].to_numpy()
    order = np.argsort(-probability_values, axis=1)

    output["primary_pick"] = [
        OUTCOMES[index] for index in order[:, 0]
    ]
    output["secondary_pick"] = [
        OUTCOMES[index] for index in order[:, 1]
    ]
    output["top_probability"] = np.max(probability_values, axis=1)
    output["probability_margin"] = (
        np.sort(probability_values, axis=1)[:, -1]
        - np.sort(probability_values, axis=1)[:, -2]
    )

    if all(column in output.columns for column in MARKET_COLUMNS.values()):
        for outcome in OUTCOMES:
            suffix = {"A": "away", "D": "draw", "H": "home"}[outcome]
            output[f"edge_{suffix}"] = (
                output[MODEL_COLUMNS[outcome]]
                - output[MARKET_COLUMNS[outcome]]
            )
            output[f"value_ratio_{suffix}"] = (
                output[MODEL_COLUMNS[outcome]]
                / output[MARKET_COLUMNS[outcome]].clip(lower=1e-15)
            )

    return output


def build_portfolio_summary(
    investment_yen: int,
    selected: list[TicketCandidate],
    reason: str,
    frame: pd.DataFrame,
    market_available: bool,
    config: OptimizerConfig,
) -> PortfolioSummary:
    estimated_ev = portfolio_ev_ratio(selected)
    expected_return = (
        investment_yen * estimated_ev
        if estimated_ev is not None
        else None
    )
    expected_profit = (
        expected_return - investment_yen
        if expected_return is not None
        else None
    )

    return PortfolioSummary(
        round_id=config.round_id,
        decision="BUY" if investment_yen > 0 else "SKIP",
        reason=reason,
        investment_yen=investment_yen,
        ticket_count=len(selected),
        available_budget_yen=available_budget(config),
        normal_budget_yen=config.normal_budget_yen,
        rollover_before_yen=config.rollover_balance_yen,
        rollover_after_yen=rollover_after_purchase(
            investment_yen,
            config,
        ),
        estimated_portfolio_ev_ratio=estimated_ev,
        estimated_expected_return_yen=expected_return,
        estimated_expected_profit_yen=expected_profit,
        model_coverage_probability=float(
            sum(ticket.model_probability for ticket in selected)
        ),
        market_coverage_probability=(
            float(
                sum(
                    ticket.market_probability or 0.0
                    for ticket in selected
                )
            )
            if market_available
            else None
        ),
        mean_ticket_confidence=float(frame["confidence_score"].mean()),
        mean_model_disagreement=float(
            frame["model_disagreement_js"].mean()
        ),
        market_data_available=market_available,
    )


def save_outputs(
    frame: pd.DataFrame,
    selected: list[TicketCandidate],
    candidates: list[TicketCandidate],
    summary: PortfolioSummary,
    config: OptimizerConfig,
) -> None:
    """Persist tickets, diagnostics, candidate ranking and state suggestion."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    ticket_frame = ticket_rows(selected, frame, config)
    ticket_frame.to_csv(
        config.output_dir / "toto_v6_tickets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    match_analysis_rows(frame).to_csv(
        config.output_dir / "toto_v6_match_analysis.csv",
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
                candidates[: min(500, len(candidates))],
                start=1,
            )
        ]
    ).to_csv(
        config.output_dir / "toto_v6_candidate_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([asdict(summary)]).to_csv(
        config.output_dir / "toto_v6_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        config.output_dir / "toto_v6_summary.json"
    ).write_text(
        json.dumps(
            asdict(summary),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    state = {
        "round_id": config.round_id,
        "rollover_balance_yen": summary.rollover_after_yen,
        "last_decision": summary.decision,
        "last_investment_yen": summary.investment_yen,
    }
    (
        config.output_dir / "rollover_state_next.json"
    ).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(
    summary: PortfolioSummary,
    config: OptimizerConfig,
) -> None:
    def optional(value: float | None, digits: int = 6) -> str:
        return "N/A" if value is None else f"{value:.{digits}f}"

    print("=" * 84)
    print("Project Alpha Toto Optimizer v6")
    print("=" * 84)
    print(f"Round ID                     : {summary.round_id}")
    print(f"Decision                     : {summary.decision}")
    print(f"Reason                       : {summary.reason}")
    print(f"Available budget             : {summary.available_budget_yen:,} yen")
    print(f"Investment                   : {summary.investment_yen:,} yen")
    print(f"Tickets                      : {summary.ticket_count}")
    print(
        f"Estimated portfolio EV       : "
        f"{optional(summary.estimated_portfolio_ev_ratio)}"
    )
    print(
        f"Estimated expected return    : "
        f"{optional(summary.estimated_expected_return_yen, 0)} yen"
    )
    print(
        f"Estimated expected profit    : "
        f"{optional(summary.estimated_expected_profit_yen, 0)} yen"
    )
    print(
        f"Model coverage probability   : "
        f"{summary.model_coverage_probability:.8f}"
    )
    print(
        f"Market coverage probability  : "
        f"{optional(summary.market_coverage_probability, 8)}"
    )
    print(f"Rollover before              : {summary.rollover_before_yen:,} yen")
    print(f"Rollover after               : {summary.rollover_after_yen:,} yen")
    print(f"Market data available        : {summary.market_data_available}")
    print(f"Outputs                      : {config.output_dir}")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Optimize up to 100 toto tickets with EV-based skip logic "
            "and rollover-aware budget control."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Round-specific prediction CSV containing exactly 13 matches.",
    )
    parser.add_argument(
        "--market-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "ml" / "toto_optimizer_v6",
    )
    parser.add_argument("--round-id", default="current")
    parser.add_argument(
        "--rollover-balance",
        type=int,
        default=0,
        help="Unused prior-round budget available for this round.",
    )
    parser.add_argument(
        "--minimum-ev",
        type=float,
        default=1.10,
    )
    parser.add_argument(
        "--allow-without-market",
        action="store_true",
        help=(
            "Allow exploratory tickets without market probabilities. "
            "Not recommended for ROI production use."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    config = OptimizerConfig(
        input_csv=args.input,
        output_dir=args.output_dir,
        market_csv=args.market_csv,
        round_id=str(args.round_id),
        rollover_balance_yen=args.rollover_balance,
        minimum_portfolio_ev=args.minimum_ev,
        allow_without_market=args.allow_without_market,
    )
    config.validate()

    frame = load_round_predictions(config)
    frame, market_available = attach_market_probabilities(
        frame,
        config.market_csv,
    )
    candidates = generate_candidates(
        frame=frame,
        market_available=market_available,
        config=config,
    )
    investment_yen, selected, reason = choose_investment(
        candidates=candidates,
        market_available=market_available,
        config=config,
    )
    summary = build_portfolio_summary(
        investment_yen=investment_yen,
        selected=selected,
        reason=reason,
        frame=frame,
        market_available=market_available,
        config=config,
    )
    save_outputs(
        frame=frame,
        selected=selected,
        candidates=candidates,
        summary=summary,
        config=config,
    )
    print_summary(summary, config)


if __name__ == "__main__":
    main()
