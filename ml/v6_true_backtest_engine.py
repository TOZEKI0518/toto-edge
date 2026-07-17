from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

LOGGER = logging.getLogger("v6_true_backtest_engine")

CLASS_ORDER = ("A", "D", "H")
PROBABILITY_COLUMNS = {
    "A": "prob_away",
    "D": "prob_draw",
    "H": "prob_home",
}
TICKET_PRICE_YEN = 100


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Project Alpha v6 true-backtest configuration."""

    rf_predictions_csv: Path
    lgbm_predictions_csv: Path
    output_dir: Path
    round_from: int = 1618
    round_to: int = 1637
    rf_weight: float = 0.60
    ticket_count: int = 50
    beam_width: int = 30_000
    candidate_limit: int = 8_000
    diversity_penalty: float = 0.10
    timeout_seconds: float = 30.0

    def validate(self) -> None:
        for label, path in (
            ("RF predictions", self.rf_predictions_csv),
            ("LightGBM predictions", self.lgbm_predictions_csv),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
            if not path.is_file():
                raise ValueError(f"{label} is not a file: {path}")

        if not 0.0 <= self.rf_weight <= 1.0:
            raise ValueError("rf_weight must be between 0 and 1.")
        if self.ticket_count <= 0:
            raise ValueError("ticket_count must be positive.")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        if self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive.")
        if self.round_from <= 0 or self.round_to <= 0:
            raise ValueError("round numbers must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")


@dataclass(frozen=True, slots=True)
class TicketCandidate:
    """One exact ticket candidate."""

    picks: str
    model_probability: float
    search_score: float


@dataclass(slots=True)
class BeamState:
    """Partial ticket candidate used in beam search."""

    picks: str
    log_probability: float
    search_score: float


class TrueBacktestError(RuntimeError):
    """Raised when the true-backtest cannot be completed safely."""


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_env_local(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding PowerShell variables."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def default_paths() -> dict[str, Path]:
    root = project_root()
    return {
        "rf": root / "ml" / "rf_v6" / "randomforest_v6_predictions.csv",
        "lgbm": root / "ml" / "lgbm_v6" / "lgbm_v6_predictions.csv",
        "output": root / "ml" / "true_backtest_v6",
    }


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.strip().lower()
    text = re.sub(r"[\s・･\-‐‑‒–—―_/／()（）\[\]［］]+", "", text)

    replacements = {
        "fc": "",
        "ＦＣ": "",
        "sc": "",
        "サッカークラブ": "",
    }
    for source, target in replacements.items():
        text = text.replace(source.lower(), target)

    aliases = {
        "横浜fマリノス": "横浜マリノス",
        "横浜fm": "横浜マリノス",
        "横浜f・マリノス": "横浜マリノス",
        "川崎フロンターレ": "川崎",
        "浦和レッズ": "浦和",
        "鹿島アントラーズ": "鹿島",
        "柏レイソル": "柏",
        "アルビレックス新潟": "新潟",
        "サンフレッチェ広島": "広島",
        "ガンバ大阪": "g大阪",
        "セレッソ大阪": "c大阪",
        "東京ヴェルディ": "東京v",
        "名古屋グランパス": "名古屋",
        "京都サンガ": "京都",
        "ヴィッセル神戸": "神戸",
        "アビスパ福岡": "福岡",
        "湘南ベルマーレ": "湘南",
        "ファジアーノ岡山": "岡山",
        "清水エスパルス": "清水",
        "fc町田ゼルビア": "町田",
        "町田ゼルビア": "町田",
    }
    return aliases.get(text, text)


def normalize_outcome(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().upper()
    direct = {
        "A": "A",
        "D": "D",
        "H": "H",
        "0": "D",
        "1": "H",
        "2": "A",
        "HOME": "H",
        "DRAW": "D",
        "AWAY": "A",
    }
    if text in direct:
        return direct[text]

    compact = re.sub(r"\s+", "", text)
    if compact in {"1", "1-0", "1ー0"}:
        return "H"
    if compact in {"0", "0-0", "0ー0"}:
        return "D"
    if compact in {"2", "0-1", "0ー1"}:
        return "A"

    return None


def normalize_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(PROBABILITY_COLUMNS.values())
    values = frame[columns].apply(pd.to_numeric, errors="coerce")

    if values.isna().any().any():
        raise TrueBacktestError(
            "Prediction probabilities contain missing or non-numeric values."
        )
    if (values < 0).any().any():
        raise TrueBacktestError(
            "Prediction probabilities contain negative values."
        )

    totals = values.sum(axis=1)
    if (totals <= 0).any():
        raise TrueBacktestError(
            "Prediction probabilities contain zero-sum rows."
        )

    output = frame.copy()
    output[columns] = values.div(totals, axis=0)
    return output


def load_prediction_file(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)

    required = {
        "match_card_id",
        "match_date",
        "home_team",
        "away_team",
        "actual",
        "prediction",
        *PROBABILITY_COLUMNS.values(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TrueBacktestError(
            f"{label} predictions missing columns: {missing}"
        )

    frame = frame.copy()
    frame["match_date"] = pd.to_datetime(
        frame["match_date"],
        errors="coerce",
    )
    frame["actual"] = frame["actual"].map(normalize_outcome)
    frame["prediction"] = frame["prediction"].map(normalize_outcome)
    frame["home_key"] = frame["home_team"].map(clean_text)
    frame["away_key"] = frame["away_team"].map(clean_text)

    frame = frame.loc[
        frame["match_date"].notna()
        & frame["actual"].isin(CLASS_ORDER)
        & frame["prediction"].isin(CLASS_ORDER)
        & frame["home_key"].ne("")
        & frame["away_key"].ne("")
    ].copy()

    if frame["match_card_id"].duplicated().any():
        duplicates = int(frame["match_card_id"].duplicated().sum())
        raise TrueBacktestError(
            f"{label} predictions contain {duplicates} duplicate match IDs."
        )

    return normalize_probabilities(frame)


def build_ensemble_predictions(
    rf: pd.DataFrame,
    lgbm: pd.DataFrame,
    rf_weight: float,
) -> pd.DataFrame:
    join_columns = [
        "match_card_id",
        "match_date",
        "home_team",
        "away_team",
        "actual",
        "home_key",
        "away_key",
    ]
    probability_columns = list(PROBABILITY_COLUMNS.values())

    rf_subset = rf[
        join_columns + probability_columns + ["prediction"]
    ].rename(
        columns={
            **{
                column: f"rf_{column}"
                for column in probability_columns
            },
            "prediction": "rf_prediction",
        }
    )
    lgbm_subset = lgbm[
        ["match_card_id", *probability_columns, "prediction"]
    ].rename(
        columns={
            **{
                column: f"lgbm_{column}"
                for column in probability_columns
            },
            "prediction": "lgbm_prediction",
        }
    )

    merged = rf_subset.merge(
        lgbm_subset,
        on="match_card_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise TrueBacktestError(
            "RF and LightGBM prediction files have no matching rows."
        )

    lgbm_weight = 1.0 - rf_weight
    for outcome, column in PROBABILITY_COLUMNS.items():
        merged[column] = (
            rf_weight * merged[f"rf_{column}"]
            + lgbm_weight * merged[f"lgbm_{column}"]
        )

    probability_values = merged[
        list(PROBABILITY_COLUMNS.values())
    ].to_numpy(dtype=float)
    indices = probability_values.argmax(axis=1)
    merged["ensemble_prediction"] = [
        CLASS_ORDER[index] for index in indices
    ]
    merged["rf_hit"] = (
        merged["rf_prediction"] == merged["actual"]
    )
    merged["lgbm_hit"] = (
        merged["lgbm_prediction"] == merged["actual"]
    )
    merged["ensemble_hit"] = (
        merged["ensemble_prediction"] == merged["actual"]
    )
    merged["models_agree"] = (
        merged["rf_prediction"] == merged["lgbm_prediction"]
    )
    return merged


class SupabaseFixtureRepository:
    """Read historical toto fixtures from Supabase REST."""

    def __init__(
        self,
        supabase_url: str,
        api_key: str,
        timeout_seconds: float,
    ) -> None:
        self.base_url = supabase_url.rstrip("/") + "/rest/v1"
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "apikey": api_key,
                "Authorization": f"Bearer {api_key}",
            }
        )

    def load_rounds(
        self,
        round_numbers: Iterable[int],
    ) -> pd.DataFrame:
        rounds = sorted(set(int(value) for value in round_numbers))
        if not rounds:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for start in range(0, len(rounds), 80):
            chunk = rounds[start : start + 80]
            response = self.session.get(
                f"{self.base_url}/toto_round_fixtures",
                params={
                    "select": (
                        "round_no,match_no,home_team,away_team,"
                        "kickoff_at,venue,toto_result"
                    ),
                    "round_no": f"in.({','.join(map(str, chunk))})",
                    "order": "round_no.asc,match_no.asc",
                },
                timeout=self.timeout_seconds,
            )
            if response.status_code != 200:
                raise TrueBacktestError(
                    "Supabase fixture fetch failed: "
                    f"status={response.status_code} body={response.text}"
                )
            payload = response.json()
            if not isinstance(payload, list):
                raise TrueBacktestError(
                    f"Unexpected Supabase fixture response: {payload}"
                )
            rows.extend(payload)

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise TrueBacktestError(
                "No toto_round_fixtures rows were returned from Supabase."
            )

        frame["round_no"] = pd.to_numeric(
            frame["round_no"],
            errors="coerce",
        ).astype("Int64")
        frame["match_no"] = pd.to_numeric(
            frame["match_no"],
            errors="coerce",
        ).astype("Int64")
        raw_kickoff = frame["kickoff_at"].copy()
        frame["kickoff_at"] = pd.to_datetime(
            raw_kickoff,
            errors="coerce",
            format="mixed",
        )

        # Supabaseの過去totoキャッシュには "03/21" のように年がない値があり、
        # pandasでは西暦1年として解釈されることがあります。
        # 年が欠落している値は、v6予測期間の最新年へ補正します。
        invalid_year = (
            frame["kickoff_at"].notna()
            & (frame["kickoff_at"].dt.year < 2000)
        )
        if invalid_year.any():
            inferred_year = 2025
            repaired: list[pd.Timestamp | pd.NaT] = []

            for raw_value, parsed_value, needs_repair in zip(
                raw_kickoff,
                frame["kickoff_at"],
                invalid_year,
                strict=True,
            ):
                if not needs_repair:
                    repaired.append(parsed_value)
                    continue

                text_value = str(raw_value).strip()
                month_day = pd.to_datetime(
                    text_value,
                    format="%m/%d",
                    errors="coerce",
                )
                if pd.isna(month_day):
                    repaired.append(pd.NaT)
                else:
                    repaired.append(
                        pd.Timestamp(
                            year=inferred_year,
                            month=int(month_day.month),
                            day=int(month_day.day),
                        )
                    )

            frame.loc[invalid_year, "kickoff_at"] = [
                value
                for value, needs_repair in zip(
                    repaired,
                    invalid_year,
                    strict=True,
                )
                if needs_repair
            ]
        frame["actual"] = frame["toto_result"].map(normalize_outcome)
        frame["home_key"] = frame["home_team"].map(clean_text)
        frame["away_key"] = frame["away_team"].map(clean_text)

        return frame.loc[
            frame["round_no"].notna()
            & frame["match_no"].notna()
            & frame["actual"].isin(CLASS_ORDER)
        ].copy()


def match_predictions_to_fixtures(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_lookup: dict[tuple[str, str], list[int]] = {}
    for index, row in predictions.iterrows():
        key = (str(row["home_key"]), str(row["away_key"]))
        prediction_lookup.setdefault(key, []).append(index)

    matched_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []

    for fixture in fixtures.itertuples(index=False):
        key = (
            clean_text(getattr(fixture, "home_team")),
            clean_text(getattr(fixture, "away_team")),
        )
        candidates = prediction_lookup.get(key, [])

        if not candidates:
            unmatched_rows.append(
                {
                    "round_no": int(fixture.round_no),
                    "match_no": int(fixture.match_no),
                    "home_team": fixture.home_team,
                    "away_team": fixture.away_team,
                    "reason": "no_team_match",
                }
            )
            continue

        candidate_frame = predictions.loc[candidates].copy()
        kickoff = getattr(fixture, "kickoff_at")

        if pd.notna(kickoff):
            kickoff_timestamp = pd.Timestamp(kickoff)

            if kickoff_timestamp.year >= 2000:
                candidate_frame["date_distance"] = (
                    candidate_frame["match_date"].dt.normalize()
                    - kickoff_timestamp.normalize()
                ).abs().dt.days
            else:
                # 年がない場合は月日だけで距離を比較します。
                target_day = pd.Timestamp(
                    year=2000,
                    month=kickoff_timestamp.month,
                    day=kickoff_timestamp.day,
                )
                candidate_days = candidate_frame["match_date"].map(
                    lambda value: pd.Timestamp(
                        year=2000,
                        month=value.month,
                        day=value.day,
                    )
                )
                candidate_frame["date_distance"] = (
                    candidate_days - target_day
                ).abs().dt.days

            candidate_frame = candidate_frame.sort_values(
                ["date_distance", "match_date"],
                kind="stable",
            )
            best = candidate_frame.iloc[0]

            if int(best["date_distance"]) > 10:
                unmatched_rows.append(
                    {
                        "round_no": int(fixture.round_no),
                        "match_no": int(fixture.match_no),
                        "home_team": fixture.home_team,
                        "away_team": fixture.away_team,
                        "reason": "date_distance_gt_10_days",
                    }
                )
                continue
        else:
            best = candidate_frame.sort_values(
                "match_date",
                ascending=False,
                kind="stable",
            ).iloc[0]

        row = best.to_dict()
        row.update(
            {
                "round_no": int(fixture.round_no),
                "toto_match_no": int(fixture.match_no),
                "fixture_home_team": fixture.home_team,
                "fixture_away_team": fixture.away_team,
                "fixture_actual": fixture.actual,
                "kickoff_at": (
                    pd.Timestamp(kickoff).isoformat()
                    if pd.notna(kickoff)
                    else None
                ),
            }
        )
        matched_rows.append(row)

    return pd.DataFrame(matched_rows), pd.DataFrame(unmatched_rows)


def generate_candidates(
    probabilities: np.ndarray,
    beam_width: int,
    candidate_limit: int,
) -> list[TicketCandidate]:
    beam = [
        BeamState(
            picks="",
            log_probability=0.0,
            search_score=0.0,
        )
    ]

    for row in probabilities:
        expanded: list[BeamState] = []
        for state in beam:
            for index, outcome in enumerate(CLASS_ORDER):
                probability = float(row[index])
                log_probability = math.log(max(probability, 1e-15))
                expanded.append(
                    BeamState(
                        picks=state.picks + outcome,
                        log_probability=(
                            state.log_probability + log_probability
                        ),
                        search_score=(
                            state.search_score + log_probability
                        ),
                    )
                )

        beam = heapq.nlargest(
            beam_width,
            expanded,
            key=lambda item: item.search_score,
        )

    return [
        TicketCandidate(
            picks=state.picks,
            model_probability=math.exp(state.log_probability),
            search_score=state.search_score,
        )
        for state in heapq.nlargest(
            candidate_limit,
            beam,
            key=lambda item: item.search_score,
        )
    ]


def ticket_similarity(first: str, second: str) -> float:
    if len(first) != len(second) or not first:
        return 0.0
    return sum(
        left == right
        for left, right in zip(first, second, strict=True)
    ) / len(first)


def select_diverse_tickets(
    candidates: list[TicketCandidate],
    ticket_count: int,
    diversity_penalty: float,
) -> list[TicketCandidate]:
    selected: list[TicketCandidate] = []
    remaining = candidates.copy()

    while len(selected) < ticket_count and remaining:
        best_index = -1
        best_score = -math.inf

        for index, candidate in enumerate(remaining):
            similarity = (
                max(
                    ticket_similarity(candidate.picks, chosen.picks)
                    for chosen in selected
                )
                if selected
                else 0.0
            )
            adjusted_score = (
                candidate.search_score
                - diversity_penalty * similarity
            )
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_index = index

        if best_index < 0:
            break
        selected.append(remaining.pop(best_index))

    return selected


def evaluate_round(
    round_frame: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    round_frame = round_frame.sort_values(
        "toto_match_no",
        kind="stable",
    ).reset_index(drop=True)

    actual = round_frame["fixture_actual"].astype(str).tolist()
    probabilities = round_frame[
        ["prob_away", "prob_draw", "prob_home"]
    ].to_numpy(dtype=float)

    candidates = generate_candidates(
        probabilities=probabilities,
        beam_width=config.beam_width,
        candidate_limit=config.candidate_limit,
    )
    tickets = select_diverse_tickets(
        candidates=candidates,
        ticket_count=config.ticket_count,
        diversity_penalty=config.diversity_penalty,
    )

    ticket_rows: list[dict[str, Any]] = []
    for rank, ticket in enumerate(tickets, start=1):
        hits = sum(
            pick == result
            for pick, result in zip(
                ticket.picks,
                actual,
                strict=True,
            )
        )
        row: dict[str, Any] = {
            "round_no": int(round_frame["round_no"].iloc[0]),
            "ticket_rank": rank,
            "picks": ticket.picks,
            "actual": "".join(actual),
            "hits": hits,
            "misses": len(actual) - hits,
            "model_probability": ticket.model_probability,
            "search_score": ticket.search_score,
        }
        for index, pick in enumerate(ticket.picks, start=1):
            row[f"match_{index:02d}"] = pick
        ticket_rows.append(row)

    tickets_frame = pd.DataFrame(ticket_rows)
    max_hits = int(tickets_frame["hits"].max()) if not tickets_frame.empty else 0
    match_count = len(round_frame)

    rf_hits = int(round_frame["rf_hit"].sum())
    lgbm_hits = int(round_frame["lgbm_hit"].sum())
    ensemble_hits = int(round_frame["ensemble_hit"].sum())

    exact_count = int((tickets_frame["hits"] == match_count).sum())
    one_miss_count = int(
        (tickets_frame["hits"] == match_count - 1).sum()
    ) if match_count >= 1 else 0
    two_miss_count = int(
        (tickets_frame["hits"] == match_count - 2).sum()
    ) if match_count >= 2 else 0

    summary = {
        "round_no": int(round_frame["round_no"].iloc[0]),
        "match_count": match_count,
        "rf_hits": rf_hits,
        "rf_accuracy": rf_hits / match_count if match_count else 0.0,
        "lgbm_hits": lgbm_hits,
        "lgbm_accuracy": lgbm_hits / match_count if match_count else 0.0,
        "ensemble_hits": ensemble_hits,
        "ensemble_accuracy": (
            ensemble_hits / match_count if match_count else 0.0
        ),
        "models_agree_count": int(round_frame["models_agree"].sum()),
        "ticket_count": len(tickets_frame),
        "investment_yen": len(tickets_frame) * TICKET_PRICE_YEN,
        "best_hits": max_hits,
        "best_hit_rate": max_hits / match_count if match_count else 0.0,
        "exact_count": exact_count,
        "one_miss_count": one_miss_count,
        "two_miss_count": two_miss_count,
        "first_prize_equivalent_count": (
            exact_count if match_count == 13 else 0
        ),
        "second_prize_equivalent_count": (
            one_miss_count if match_count == 13 else 0
        ),
        "third_prize_equivalent_count": (
            two_miss_count if match_count == 13 else 0
        ),
        "monetary_roi": None,
        "monetary_roi_note": (
            "Historical payout data is not connected yet."
        ),
    }
    return summary, tickets_frame


def confidence_band(frame: pd.DataFrame) -> pd.Series:
    probabilities = frame[
        ["prob_away", "prob_draw", "prob_home"]
    ].to_numpy(dtype=float)
    sorted_values = np.sort(probabilities, axis=1)[:, ::-1]
    top = sorted_values[:, 0]
    margin = sorted_values[:, 0] - sorted_values[:, 1]

    return pd.Series(
        np.select(
            [
                (top >= 0.48) & (margin >= 0.08),
                (top >= 0.40) & (margin >= 0.03),
            ],
            ["HIGH", "MEDIUM"],
            default="LOW",
        ),
        index=frame.index,
    )


def build_overall_summary(
    matches: pd.DataFrame,
    rounds: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, Any]:
    if matches.empty:
        raise TrueBacktestError("No matched v6 rows were evaluated.")

    matches = matches.copy()
    matches["confidence_level"] = confidence_band(matches)

    confidence_rows = {}
    for level in ("HIGH", "MEDIUM", "LOW"):
        subset = matches.loc[matches["confidence_level"] == level]
        confidence_rows[level] = {
            "rows": len(subset),
            "ensemble_accuracy": (
                float(subset["ensemble_hit"].mean())
                if not subset.empty
                else None
            ),
        }

    return {
        "round_from": config.round_from,
        "round_to": config.round_to,
        "rounds_evaluated": len(rounds),
        "matches_evaluated": len(matches),
        "rf_accuracy": float(matches["rf_hit"].mean()),
        "lgbm_accuracy": float(matches["lgbm_hit"].mean()),
        "ensemble_accuracy": float(matches["ensemble_hit"].mean()),
        "models_agree_rate": float(matches["models_agree"].mean()),
        "average_best_hits": float(rounds["best_hits"].mean()),
        "average_match_count": float(rounds["match_count"].mean()),
        "exact_ticket_rounds": int((rounds["exact_count"] > 0).sum()),
        "one_miss_or_better_rounds": int(
            (
                (rounds["exact_count"] > 0)
                | (rounds["one_miss_count"] > 0)
            ).sum()
        ),
        "two_miss_or_better_rounds": int(
            (
                (rounds["exact_count"] > 0)
                | (rounds["one_miss_count"] > 0)
                | (rounds["two_miss_count"] > 0)
            ).sum()
        ),
        "total_investment_yen": int(rounds["investment_yen"].sum()),
        "confidence": confidence_rows,
        "monetary_roi": None,
        "monetary_roi_note": (
            "Historical payout data is not connected yet. "
            "Accuracy and 50-ticket hit counts are true out-of-sample v6 results."
        ),
    }


def parse_args() -> argparse.Namespace:
    defaults = default_paths()

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Project Alpha v6 out-of-sample RF/LightGBM "
            "predictions by historical toto round and generate 50 tickets."
        )
    )
    parser.add_argument(
        "--rf-predictions",
        type=Path,
        default=defaults["rf"],
    )
    parser.add_argument(
        "--lgbm-predictions",
        type=Path,
        default=defaults["lgbm"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults["output"],
    )
    parser.add_argument("--round-from", type=int, default=1618)
    parser.add_argument("--round-to", type=int, default=1637)
    parser.add_argument("--rf-weight", type=float, default=0.60)
    parser.add_argument("--ticket-count", type=int, default=50)
    parser.add_argument("--beam-width", type=int, default=30_000)
    parser.add_argument("--candidate-limit", type=int, default=8_000)
    parser.add_argument("--diversity-penalty", type=float, default=0.10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_env_local(project_root() / ".env.local")

    args = parse_args()
    configure_logging(args.verbose)

    config = BacktestConfig(
        rf_predictions_csv=args.rf_predictions,
        lgbm_predictions_csv=args.lgbm_predictions,
        output_dir=args.output_dir,
        round_from=min(args.round_from, args.round_to),
        round_to=max(args.round_from, args.round_to),
        rf_weight=args.rf_weight,
        ticket_count=args.ticket_count,
        beam_width=args.beam_width,
        candidate_limit=args.candidate_limit,
        diversity_penalty=args.diversity_penalty,
        timeout_seconds=args.timeout,
    )
    config.validate()

    supabase_url = (
        os.getenv("SUPABASE_URL")
        or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        or ""
    ).strip()
    api_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SECRET_KEY")
        or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        or ""
    ).strip()

    if not supabase_url:
        raise TrueBacktestError(
            "SUPABASE_URL or NEXT_PUBLIC_SUPABASE_URL is not set."
        )
    if not api_key:
        raise TrueBacktestError(
            "A Supabase API key environment variable is not set."
        )

    LOGGER.info("Loading RF v6 out-of-sample predictions.")
    rf = load_prediction_file(config.rf_predictions_csv, "RF")
    LOGGER.info("Loading LightGBM v6 out-of-sample predictions.")
    lgbm = load_prediction_file(config.lgbm_predictions_csv, "LightGBM")
    predictions = build_ensemble_predictions(
        rf=rf,
        lgbm=lgbm,
        rf_weight=config.rf_weight,
    )

    repository = SupabaseFixtureRepository(
        supabase_url=supabase_url,
        api_key=api_key,
        timeout_seconds=config.timeout_seconds,
    )
    fixtures = repository.load_rounds(
        range(config.round_from, config.round_to + 1)
    )
    matched, unmatched = match_predictions_to_fixtures(
        fixtures=fixtures,
        predictions=predictions,
    )

    if matched.empty:
        raise TrueBacktestError(
            "No historical toto fixtures matched v6 prediction rows."
        )

    round_summaries: list[dict[str, Any]] = []
    ticket_parts: list[pd.DataFrame] = []

    for round_no, round_frame in matched.groupby(
        "round_no",
        sort=True,
    ):
        summary, tickets = evaluate_round(
            round_frame=round_frame,
            config=config,
        )
        round_summaries.append(summary)
        ticket_parts.append(tickets)
        LOGGER.info(
            "Round %d | matches=%d ensemble=%d/%d best_ticket=%d/%d",
            round_no,
            summary["match_count"],
            summary["ensemble_hits"],
            summary["match_count"],
            summary["best_hits"],
            summary["match_count"],
        )

    rounds = pd.DataFrame(round_summaries).sort_values(
        "round_no",
        ascending=False,
        kind="stable",
    )
    tickets = (
        pd.concat(ticket_parts, ignore_index=True)
        if ticket_parts
        else pd.DataFrame()
    )
    matched = matched.sort_values(
        ["round_no", "toto_match_no"],
        kind="stable",
    )
    overall = build_overall_summary(
        matches=matched,
        rounds=rounds,
        config=config,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rounds.to_csv(
        config.output_dir / "v6_true_backtest_rounds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tickets.to_csv(
        config.output_dir / "v6_true_backtest_tickets.csv",
        index=False,
        encoding="utf-8-sig",
    )
    matched.to_csv(
        config.output_dir / "v6_true_backtest_matches.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unmatched.to_csv(
        config.output_dir / "v6_true_backtest_unmatched.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        config.output_dir / "v6_true_backtest_summary.json"
    ).write_text(
        json.dumps(
            overall,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame([overall]).to_csv(
        config.output_dir / "v6_true_backtest_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 92)
    print("Project Alpha v6 True Backtest Engine")
    print("=" * 92)
    print(f"Rounds evaluated           : {overall['rounds_evaluated']}")
    print(f"Matches evaluated          : {overall['matches_evaluated']}")
    print(f"RF accuracy                : {overall['rf_accuracy']:.4f}")
    print(f"LightGBM accuracy          : {overall['lgbm_accuracy']:.4f}")
    print(f"Ensemble accuracy          : {overall['ensemble_accuracy']:.4f}")
    print(f"Average best ticket hits   : {overall['average_best_hits']:.4f}")
    print(f"Exact-ticket rounds        : {overall['exact_ticket_rounds']}")
    print(
        "One-miss-or-better rounds : "
        f"{overall['one_miss_or_better_rounds']}"
    )
    print(
        "Two-miss-or-better rounds : "
        f"{overall['two_miss_or_better_rounds']}"
    )
    print(f"Total investment           : {overall['total_investment_yen']:,} yen")
    print(f"Output                     : {config.output_dir}")


if __name__ == "__main__":
    main()
