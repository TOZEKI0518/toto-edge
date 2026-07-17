from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from ml.confidence_engine import ConfidenceEngineBuilder

LOGGER: Final[logging.Logger] = logging.getLogger(
    "current_round_confidence_optimizer_v6"
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
RF_COLUMNS: Final[dict[str, str]] = {
    "A": "rf_prob_away",
    "D": "rf_prob_draw",
    "H": "rf_prob_home",
}
LGBM_COLUMNS: Final[dict[str, str]] = {
    "A": "lgbm_prob_away",
    "D": "lgbm_prob_draw",
    "H": "lgbm_prob_home",
}


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Confidence Engine V3 and ROI optimizer configuration."""

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
        """Validate optimizer settings and input path."""
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
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        if self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive.")


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
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def project_root() -> Path:
    """Return repository root for this script under ml/."""
    return Path(__file__).resolve().parents[1]


def _normalize_probability_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    """Validate and normalize one probability group."""
    values = frame[columns].apply(pd.to_numeric, errors="coerce")

    if values.isna().any().any():
        raise ConfidenceOptimizerError(
            f"{label} probabilities contain missing/non-numeric values."
        )
    if (values < 0.0).any().any():
        raise ConfidenceOptimizerError(
            f"{label} probabilities contain negative values."
        )

    totals = values.sum(axis=1)
    if (totals <= 0.0).any():
        raise ConfidenceOptimizerError(
            f"{label} probabilities contain zero-sum rows."
        )

    return values.div(totals, axis=0)


def load_input(path: Path) -> pd.DataFrame:
    """Load, validate, normalize, and sort current-round data."""
    frame = pd.read_csv(path, encoding="utf-8-sig")

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

    expected_match_numbers = list(range(1, 14))
    actual_match_numbers = (
        frame["toto_match_no"].astype(int).tolist()
    )
    if actual_match_numbers != expected_match_numbers:
        raise ConfidenceOptimizerError(
            "Expected toto_match_no 1..13, found "
            f"{actual_match_numbers}."
        )

    if frame["match_card_id"].duplicated().any():
        raise ConfidenceOptimizerError(
            "Duplicate match_card_id values found."
        )

    frame[list(AI_COLUMNS.values())] = (
        _normalize_probability_columns(
            frame,
            list(AI_COLUMNS.values()),
            label="AI",
        )
    )
    frame[list(MARKET_COLUMNS.values())] = (
        _normalize_probability_columns(
            frame,
            list(MARKET_COLUMNS.values()),
            label="Market",
        )
    )

    for columns, label in (
        (RF_COLUMNS, "Random Forest"),
        (LGBM_COLUMNS, "LightGBM"),
    ):
        available = [
            column in frame.columns for column in columns.values()
        ]
        if any(available) and not all(available):
            missing_model_columns = [
                column
                for column in columns.values()
                if column not in frame.columns
            ]
            raise ConfidenceOptimizerError(
                f"{label} probability group is incomplete: "
                + ", ".join(missing_model_columns)
            )
        if all(available):
            frame[list(columns.values())] = (
                _normalize_probability_columns(
                    frame,
                    list(columns.values()),
                    label=label,
                )
            )

    return frame


def _build_engine_prediction_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert legacy current-round columns to Confidence Engine V3 schema.

    When RF or LightGBM probabilities are unavailable, ensemble probabilities
    are used as a neutral fallback. This preserves compatibility without
    fabricating model disagreement.
    """
    prediction = pd.DataFrame(
        {
            "match_id": frame["match_card_id"].astype(str),
            "home_team": frame["home_team"].astype(str),
            "away_team": frame["away_team"].astype(str),
            "ensemble_prob_a": frame[AI_COLUMNS["A"]],
            "ensemble_prob_d": frame[AI_COLUMNS["D"]],
            "ensemble_prob_h": frame[AI_COLUMNS["H"]],
        }
    )

    for outcome, suffix in (("A", "a"), ("D", "d"), ("H", "h")):
        rf_source = (
            RF_COLUMNS[outcome]
            if RF_COLUMNS[outcome] in frame.columns
            else AI_COLUMNS[outcome]
        )
        lgbm_source = (
            LGBM_COLUMNS[outcome]
            if LGBM_COLUMNS[outcome] in frame.columns
            else AI_COLUMNS[outcome]
        )
        prediction[f"rf_prob_{suffix}"] = frame[rf_source]
        prediction[f"lgbm_prob_{suffix}"] = frame[lgbm_source]

    return prediction


def _build_engine_market_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Convert legacy market columns to Confidence Engine V3 schema."""
    return pd.DataFrame(
        {
            "match_id": frame["match_card_id"].astype(str),
            "home_team": frame["home_team"].astype(str),
            "away_team": frame["away_team"].astype(str),
            "market_prob_a": frame[MARKET_COLUMNS["A"]],
            "market_prob_d": frame[MARKET_COLUMNS["D"]],
            "market_prob_h": frame[MARKET_COLUMNS["H"]],
        }
    )


def build_confidence_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Generate confidence features through Confidence Engine Version 3.

    Legacy output aliases are retained because the current beam-search and
    output files still use the Version 6 column names.
    """
    prediction_frame = _build_engine_prediction_frame(frame)
    market_frame = _build_engine_market_frame(frame)

    engine = ConfidenceEngineBuilder()
    run_result = engine.build_from_frames(
        predictions=prediction_frame,
        market=market_frame,
        metadata={
            "integration": "current_round_confidence_optimizer_v6",
            "round_id": int(frame["round_id"].iloc[0]),
        },
    )

    engine_output = run_result.dataframe.copy()
    engine_output = engine_output.rename(
        columns={
            "match_id": "match_card_id",
            "top_outcome": "primary_pick",
            "second_outcome": "secondary_pick",
            "normalized_entropy": "probability_entropy",
            "entropy_certainty": "certainty_from_entropy",
            "model_js_distance": "model_disagreement_js",
            "model_same_top_pick": "models_agree_top_pick",
            "ticket_recommendation": "coverage_recommendation",
            "market_best_edge": "best_edge",
            "market_best_value_ratio": "best_value_ratio",
        }
    )

    selected_columns = [
        "match_card_id",
        "primary_pick",
        "secondary_pick",
        "top_probability",
        "second_probability",
        "third_probability",
        "probability_margin",
        "probability_entropy",
        "certainty_from_entropy",
        "model_disagreement_js",
        "models_agree_top_pick",
        "model_max_probability_gap",
        "confidence_score",
        "confidence_level",
        "coverage_recommendation",
        "coverage_count",
        "covered_outcomes",
        "best_edge",
        "best_value_ratio",
        "market_edge_level",
        "roi_priority_score",
        "ai_market_same_top_pick",
        "market_edge_for_ai_pick",
        "market_value_ratio_for_ai_pick",
    ]
    available_columns = [
        column
        for column in selected_columns
        if column in engine_output.columns
    ]

    output = frame.merge(
        engine_output.loc[:, available_columns],
        on="match_card_id",
        how="left",
        validate="one_to_one",
    )

    if output["confidence_score"].isna().any():
        raise ConfidenceOptimizerError(
            "Confidence Engine V3 output could not be matched to all rows."
        )

    output["recommended_combination"] = (
        output["covered_outcomes"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    output["uncertainty_score"] = (
        1.0 - output["confidence_score"]
    ).clip(0.0, 1.0)

    LOGGER.info(
        "Confidence Engine V3 integrated: rows=%d, high=%d, "
        "medium=%d, low=%d",
        len(output),
        int((output["confidence_level"] == "HIGH").sum()),
        int((output["confidence_level"] == "MEDIUM").sum()),
        int((output["confidence_level"] == "LOW").sum()),
    )

    return output


def available_budget(config: OptimizerConfig) -> int:
    """Calculate currently usable budget."""
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
    roi_priority: float,
    config: OptimizerConfig,
) -> float:
    """Calculate one beam-search option score."""
    value_ratio = model_probability / max(
        market_probability,
        1e-15,
    )
    return (
        math.log(max(model_probability, 1e-15))
        + config.value_weight
        * math.log(max(value_ratio, 1e-15))
        + config.confidence_weight * confidence
        + 0.10 * max(roi_priority, 0.0)
        - config.disagreement_penalty * disagreement
    )


def generate_candidates(
    frame: pd.DataFrame,
    config: OptimizerConfig,
) -> list[TicketCandidate]:
    """Generate candidate tickets through beam search."""
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
                    roi_priority=float(
                        row.get("roi_priority_score", 0.0)
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
    final_states = heapq.nlargest(
        config.candidate_limit,
        beam,
        key=lambda state: state.search_score,
    )

    for state in final_states:
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
                min(
                    max(ratio, 1e-12),
                    config.per_match_value_cap,
                )
            )

        conservative_value_index = float(
            math.exp(
                sum(
                    math.log(value)
                    for value in leg_value_ratios
                )
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
    """Return the share of identical picks between two tickets."""
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
    """Select high-scoring tickets while penalizing duplication."""
    selected: list[TicketCandidate] = []
    remaining = candidates.copy()

    while len(selected) < ticket_count and remaining:
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
            adjusted_score = (
                candidate.search_score
                - config.diversity_penalty * max_similarity
            )
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_index = index

        if best_index < 0:
            break

        selected.append(remaining.pop(best_index))

    return selected


def budget_levels(
    available_yen: int,
    ticket_price_yen: int,
) -> list[int]:
    """Build supported investment levels."""
    standard_levels = (
        10_000,
        7_500,
        5_000,
        3_000,
        2_000,
        1_000,
    )
    levels = {
        amount
        for amount in standard_levels
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
    """Return mean conservative value index for selected tickets."""
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
    """Choose the largest acceptable portfolio within budget."""
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
                    "Conservative portfolio value index "
                    f"{estimated_value:.4f} >= "
                    f"{config.minimum_portfolio_value_index:.4f}."
                ),
            )

    return (
        0,
        [],
        (
            "No candidate budget maintained the minimum "
            "portfolio value index of "
            f"{config.minimum_portfolio_value_index:.4f}."
        ),
    )


def rollover_after(
    investment_yen: int,
    config: OptimizerConfig,
) -> int:
    """Calculate rollover balance after the current decision."""
    rollover_before = min(
        config.rollover_balance_yen,
        config.rollover_cap_yen,
    )

    if investment_yen <= config.normal_budget_yen:
        unused_normal_budget = (
            config.normal_budget_yen - investment_yen
        )
        return min(
            config.rollover_cap_yen,
            rollover_before + unused_normal_budget,
        )

    used_rollover = (
        investment_yen - config.normal_budget_yen
    )
    return max(0, rollover_before - used_rollover)


def build_ticket_frame(
    tickets: list[TicketCandidate],
    round_id: int,
    ticket_price_yen: int,
) -> pd.DataFrame:
    """Convert selected tickets to an export-ready DataFrame."""
    rows: list[dict[str, Any]] = []

    for ticket_number, ticket in enumerate(tickets, start=1):
        row: dict[str, Any] = {
            "round_id": round_id,
            "ticket_number": ticket_number,
            "ticket_cost_yen": ticket_price_yen,
            "picks": ticket.picks,
            "model_probability": ticket.model_probability,
            "market_probability": ticket.market_probability,
            "conservative_value_index": (
                ticket.conservative_value_index
            ),
            "search_score": ticket.search_score,
        }

        for match_index, pick in enumerate(
            ticket.picks,
            start=1,
        ):
            row[f"match_{match_index:02d}"] = pick

        rows.append(row)

    return pd.DataFrame(rows)


def build_summary(
    frame: pd.DataFrame,
    investment: int,
    tickets: list[TicketCandidate],
    reason: str,
    config: OptimizerConfig,
) -> PortfolioSummary:
    """Build selected portfolio summary."""
    estimated_value = portfolio_value_index(tickets)

    return PortfolioSummary(
        round_id=int(frame["round_id"].iloc[0]),
        decision="BUY" if investment > 0 else "SKIP",
        reason=reason,
        investment_yen=investment,
        ticket_count=len(tickets),
        available_budget_yen=available_budget(config),
        normal_budget_yen=config.normal_budget_yen,
        rollover_before_yen=config.rollover_balance_yen,
        rollover_after_yen=rollover_after(
            investment,
            config,
        ),
        estimated_portfolio_value_index=estimated_value,
        estimated_expected_return_yen=None,
        estimated_expected_profit_yen=None,
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
    """Persist confidence, candidate, ticket, and summary outputs."""
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
    """Parse command-line arguments."""
    root = project_root()

    parser = argparse.ArgumentParser(
        description=(
            "Connect Confidence Engine V3 and the current-round "
            "ROI optimizer for Project Alpha."
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
    """CLI entry point."""
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
    print("Project Alpha Confidence Engine V3 + ROI Optimizer")
    print("=" * 92)
    print(f"Round ID                  : {summary.round_id}")
    print(f"Decision                  : {summary.decision}")
    print(f"Reason                    : {summary.reason}")
    print(
        "Available budget          : "
        f"{summary.available_budget_yen:,} yen"
    )
    print(
        "Investment                : "
        f"{summary.investment_yen:,} yen"
    )
    print(f"Ticket count              : {summary.ticket_count}")
    print(
        "Conservative portfolio value index : "
        f"{summary.estimated_portfolio_value_index}"
    )
    print(
        "High/Medium/Low matches   : "
        f"{summary.high_confidence_matches}/"
        f"{summary.medium_confidence_matches}/"
        f"{summary.low_confidence_matches}"
    )
    print(
        "Rollover before/after     : "
        f"{summary.rollover_before_yen:,}/"
        f"{summary.rollover_after_yen:,} yen"
    )
    print(f"Outputs                   : {config.output_dir}")


if __name__ == "__main__":
    main()
