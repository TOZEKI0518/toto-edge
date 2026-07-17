from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

LOGGER = logging.getLogger("supabase_current_round_uploader_v6")


@dataclass(frozen=True, slots=True)
class UploadConfig:
    """Configuration for uploading current-round Project Alpha outputs."""

    market_predictions_csv: Path
    confidence_csv: Path
    tickets_csv: Path
    optimizer_summary_json: Path
    monte_carlo_tickets_csv: Path
    monte_carlo_summary_json: Path
    timeout_seconds: float = 30.0

    def validate(self) -> None:
        for label, path in (
            ("market_predictions_csv", self.market_predictions_csv),
            ("confidence_csv", self.confidence_csv),
            ("tickets_csv", self.tickets_csv),
            ("optimizer_summary_json", self.optimizer_summary_json),
            ("monte_carlo_tickets_csv", self.monte_carlo_tickets_csv),
            ("monte_carlo_summary_json", self.monte_carlo_summary_json),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
            if not path.is_file():
                raise ValueError(f"{label} is not a file: {path}")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")


class SupabaseRestClient:
    """Minimal Supabase REST client using PostgREST."""

    def __init__(
        self,
        supabase_url: str,
        service_role_key: str,
        timeout_seconds: float,
    ) -> None:
        self.base_url = supabase_url.rstrip("/") + "/rest/v1"
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "apikey": service_role_key,
                "Authorization": f"Bearer {service_role_key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=representation",
            }
        )

    def upsert(
        self,
        table: str,
        rows: list[dict[str, Any]],
        on_conflict: str,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []

        response = self.session.post(
            f"{self.base_url}/{table}",
            params={"on_conflict": on_conflict},
            data=json.dumps(rows, ensure_ascii=False),
            timeout=self.timeout_seconds,
        )

        if response.status_code not in {200, 201}:
            raise RuntimeError(
                f"Supabase upsert failed for {table}: "
                f"status={response.status_code} body={response.text}"
            )

        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(
                f"Unexpected Supabase response for {table}: {payload}"
            )
        return payload


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def dataframe_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            str(column): clean_value(value)
            for column, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def load_market_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    required = {
        "round_id",
        "toto_match_no",
        "match_card_id",
        "match_date",
        "home_team",
        "away_team",
        "prediction",
        "prob_away",
        "prob_draw",
        "prob_home",
        "market_prob_away",
        "market_prob_draw",
        "market_prob_home",
        "best_edge",
        "best_value_ratio",
        "best_value_pick",
        "value_score",
        "roi_priority_score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Market prediction CSV is missing required columns: "
            + ", ".join(missing)
        )

    if len(frame) != 13:
        raise ValueError(
            f"Expected 13 market prediction rows, found {len(frame)}."
        )

    return frame


def load_confidence(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    required = {
        "round_id",
        "toto_match_no",
        "confidence_score",
        "confidence_level",
        "coverage_recommendation",
        "recommended_combination",
        "primary_pick",
        "secondary_pick",
        "model_disagreement_js",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Confidence CSV is missing required columns: "
            + ", ".join(missing)
        )

    if len(frame) != 13:
        raise ValueError(
            f"Expected 13 confidence rows, found {len(frame)}."
        )

    return frame


def load_tickets(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    if frame.empty:
        return frame

    required = {
        "round_id",
        "ticket_number",
        "ticket_cost_yen",
        "picks",
        "model_probability",
        "market_probability",
        "conservative_value_index",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Tickets CSV is missing required columns: "
            + ", ".join(missing)
        )

    return frame


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Optimizer summary JSON must contain an object.")

    required = {
        "round_id",
        "decision",
        "reason",
        "investment_yen",
        "ticket_count",
        "available_budget_yen",
        "normal_budget_yen",
        "rollover_before_yen",
        "rollover_after_yen",
        "estimated_portfolio_value_index",
        "model_coverage_probability",
        "market_coverage_probability",
        "high_confidence_matches",
        "medium_confidence_matches",
        "low_confidence_matches",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            "Optimizer summary JSON is missing keys: "
            + ", ".join(missing)
        )

    return payload



def load_monte_carlo_summary(path: Path) -> dict[str, Any]:
    """Load and validate Monte Carlo optimizer summary."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Monte Carlo summary JSON must contain an object.")

    required = {
        "round_id",
        "seed",
        "simulations",
        "optimization_iterations",
        "ticket_count",
        "investment_yen",
        "initial_metrics",
        "optimized_metrics",
        "objective_improvement",
        "hit_rate_improvement",
        "probability_11_plus_improvement",
        "probability_12_plus_improvement",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            "Monte Carlo summary JSON is missing keys: "
            + ", ".join(missing)
        )

    for key in ("initial_metrics", "optimized_metrics"):
        if not isinstance(payload[key], dict):
            raise ValueError(f"{key} must contain an object.")

    return payload


def build_match_rows(
    predictions: pd.DataFrame,
    confidence: pd.DataFrame,
) -> list[dict[str, Any]]:
    confidence_subset = confidence[
        [
            "round_id",
            "toto_match_no",
            "confidence_score",
            "confidence_level",
            "coverage_recommendation",
            "recommended_combination",
            "primary_pick",
            "secondary_pick",
            "model_disagreement_js",
        ]
    ].copy()

    merged = predictions.merge(
        confidence_subset,
        on=["round_id", "toto_match_no"],
        how="left",
        validate="one_to_one",
    )

    selected_columns = [
        "round_id",
        "toto_match_no",
        "match_card_id",
        "match_date",
        "home_team",
        "away_team",
        "prediction",
        "prob_away",
        "prob_draw",
        "prob_home",
        "rf_prediction",
        "rf_prob_away",
        "rf_prob_draw",
        "rf_prob_home",
        "lgbm_prediction",
        "lgbm_prob_away",
        "lgbm_prob_draw",
        "lgbm_prob_home",
        "market_prob_away",
        "market_prob_draw",
        "market_prob_home",
        "best_edge",
        "best_value_ratio",
        "best_value_pick",
        "value_score",
        "roi_priority_score",
        "models_agree",
        "ai_market_agree",
        "confidence_score",
        "confidence_level",
        "coverage_recommendation",
        "recommended_combination",
        "primary_pick",
        "secondary_pick",
        "model_disagreement_js",
    ]

    existing_columns = [
        column for column in selected_columns
        if column in merged.columns
    ]

    rows = dataframe_to_records(merged[existing_columns])

    for row in rows:
        row["round_id"] = int(row["round_id"])
        row["toto_match_no"] = int(row["toto_match_no"])
        row["match_card_id"] = int(row["match_card_id"])

    return rows


def build_ticket_rows(
    tickets: pd.DataFrame,
) -> list[dict[str, Any]]:
    if tickets.empty:
        return []

    rows = dataframe_to_records(tickets)
    for row in rows:
        row["round_id"] = int(row["round_id"])
        row["ticket_number"] = int(row["ticket_number"])
        row["ticket_cost_yen"] = int(row["ticket_cost_yen"])

    return rows


def build_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    row = dict(summary)
    row["round_id"] = int(row["round_id"])
    row["investment_yen"] = int(row["investment_yen"])
    row["ticket_count"] = int(row["ticket_count"])
    row["available_budget_yen"] = int(row["available_budget_yen"])
    row["normal_budget_yen"] = int(row["normal_budget_yen"])
    row["rollover_before_yen"] = int(row["rollover_before_yen"])
    row["rollover_after_yen"] = int(row["rollover_after_yen"])
    row["high_confidence_matches"] = int(
        row["high_confidence_matches"]
    )
    row["medium_confidence_matches"] = int(
        row["medium_confidence_matches"]
    )
    row["low_confidence_matches"] = int(
        row["low_confidence_matches"]
    )
    return {
        key: clean_value(value)
        for key, value in row.items()
    }



def build_monte_carlo_summary_row(
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Flatten Monte Carlo before/after metrics for Supabase."""
    initial = summary["initial_metrics"]
    optimized = summary["optimized_metrics"]

    row: dict[str, Any] = {
        "round_id": int(summary["round_id"]),
        "seed": int(summary["seed"]),
        "simulations": int(summary["simulations"]),
        "optimization_iterations": int(summary["optimization_iterations"]),
        "ticket_count": int(summary["ticket_count"]),
        "investment_yen": int(summary["investment_yen"]),
    }

    metric_names = (
        "exact_hit_probability",
        "simulated_hit_rate",
        "mean_best_match_count",
        "probability_10_or_more",
        "probability_11_or_more",
        "probability_12_or_more",
        "probability_13",
        "average_pair_similarity",
        "maximum_pair_similarity",
        "unique_ticket_ratio",
        "average_value_index",
        "tail_coverage_score",
        "objective_score",
    )
    for name in metric_names:
        row[f"initial_{name}"] = initial.get(name)
        row[f"optimized_{name}"] = optimized.get(name)

    for name in (
        "objective_improvement",
        "hit_rate_improvement",
        "probability_11_plus_improvement",
        "probability_12_plus_improvement",
    ):
        row[name] = summary.get(name)

    return {key: clean_value(value) for key, value in row.items()}


def parse_args() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(
        description=(
            "Upload Project Alpha current-round predictions, "
            "confidence, optimizer summary and tickets to Supabase."
        )
    )
    parser.add_argument(
        "--market-predictions",
        type=Path,
        default=(
            root
            / "ml"
            / "current_round"
            / "current_round_market_predictions_v6.csv"
        ),
    )
    parser.add_argument(
        "--confidence",
        type=Path,
        default=(
            root
            / "ml"
            / "current_round"
            / "optimizer_v6"
            / "current_round_confidence_v6.csv"
        ),
    )
    parser.add_argument(
        "--tickets",
        type=Path,
        default=(
            root
            / "ml"
            / "current_round"
            / "optimizer_v6"
            / "current_round_tickets_v6.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=(
            root
            / "ml"
            / "current_round"
            / "optimizer_v6"
            / "current_round_optimizer_v6_summary.json"
        ),
    )
    parser.add_argument(
        "--monte-carlo-tickets",
        type=Path,
        default=(
            root
            / "ml"
            / "current_round"
            / "optimizer_v6"
            / "current_round_tickets_monte_carlo.csv"
        ),
    )
    parser.add_argument(
        "--monte-carlo-summary",
        type=Path,
        default=(
            root
            / "ml"
            / "current_round"
            / "optimizer_v6"
            / "monte_carlo_optimizer_summary.json"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    config = UploadConfig(
        market_predictions_csv=args.market_predictions,
        confidence_csv=args.confidence,
        tickets_csv=args.tickets,
        optimizer_summary_json=args.summary,
        monte_carlo_tickets_csv=args.monte_carlo_tickets,
        monte_carlo_summary_json=args.monte_carlo_summary,
        timeout_seconds=args.timeout,
    )
    config.validate()

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_role_key = os.getenv(
        "SUPABASE_SERVICE_ROLE_KEY",
        "",
    ).strip()

    if not supabase_url:
        raise RuntimeError(
            "SUPABASE_URL environment variable is not set."
        )
    if not service_role_key:
        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY environment variable is not set."
        )

    predictions = load_market_predictions(
        config.market_predictions_csv
    )
    confidence = load_confidence(config.confidence_csv)
    tickets = load_tickets(config.tickets_csv)
    summary = load_summary(config.optimizer_summary_json)
    monte_carlo_tickets = load_tickets(config.monte_carlo_tickets_csv)
    monte_carlo_summary = load_monte_carlo_summary(
        config.monte_carlo_summary_json
    )

    round_ids = {
        int(predictions["round_id"].iloc[0]),
        int(confidence["round_id"].iloc[0]),
        int(summary["round_id"]),
        int(monte_carlo_summary["round_id"]),
    }
    if not tickets.empty:
        round_ids.add(int(tickets["round_id"].iloc[0]))
    if not monte_carlo_tickets.empty:
        round_ids.add(int(monte_carlo_tickets["round_id"].iloc[0]))

    if len(round_ids) != 1:
        raise RuntimeError(
            f"Round ID mismatch across files: {sorted(round_ids)}"
        )

    client = SupabaseRestClient(
        supabase_url=supabase_url,
        service_role_key=service_role_key,
        timeout_seconds=config.timeout_seconds,
    )

    match_rows = build_match_rows(
        predictions=predictions,
        confidence=confidence,
    )
    ticket_rows = build_ticket_rows(tickets)
    summary_row = build_summary_row(summary)
    monte_carlo_ticket_rows = build_ticket_rows(monte_carlo_tickets)
    monte_carlo_summary_row = build_monte_carlo_summary_row(
        monte_carlo_summary
    )

    uploaded_summary = client.upsert(
        table="toto_round_runs",
        rows=[summary_row],
        on_conflict="round_id",
    )
    uploaded_matches = client.upsert(
        table="toto_round_match_predictions",
        rows=match_rows,
        on_conflict="round_id,toto_match_no",
    )
    uploaded_tickets = client.upsert(
        table="toto_round_tickets",
        rows=ticket_rows,
        on_conflict="round_id,ticket_number",
    )
    uploaded_monte_carlo_summary = client.upsert(
        table="toto_round_monte_carlo_runs",
        rows=[monte_carlo_summary_row],
        on_conflict="round_id",
    )
    uploaded_monte_carlo_tickets = client.upsert(
        table="toto_round_monte_carlo_tickets",
        rows=monte_carlo_ticket_rows,
        on_conflict="round_id,ticket_number",
    )

    print("=" * 92)
    print("Project Alpha Supabase Current Round Uploader v6")
    print("=" * 92)
    print(f"Round ID                  : {next(iter(round_ids))}")
    print(f"Round summary rows        : {len(uploaded_summary)}")
    print(f"Match prediction rows     : {len(uploaded_matches)}")
    print(f"Standard ticket rows      : {len(uploaded_tickets)}")
    print(
        "Monte Carlo summary rows  : "
        f"{len(uploaded_monte_carlo_summary)}"
    )
    print(
        "Monte Carlo ticket rows   : "
        f"{len(uploaded_monte_carlo_tickets)}"
    )
    print("Status                    : Upload completed")


if __name__ == "__main__":
    main()
