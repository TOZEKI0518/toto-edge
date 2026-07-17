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
    date_window_days: int = 18
    minimum_round_coverage: float = 0.75
    jleague_only: bool = True

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
        if self.date_window_days <= 0:
            raise ValueError("date_window_days must be positive.")
        if not 0.0 < self.minimum_round_coverage <= 1.0:
            raise ValueError(
                "minimum_round_coverage must be between 0 and 1."
            )


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
    """Normalize Japanese club names to a stable short key."""
    if value is None or pd.isna(value):
        return ""

    raw = unicodedata.normalize("NFKC", str(value)).strip()
    compact = re.sub(
        r"[\s・･\-‐‑‒–—―_/／()（）\[\]［］.]+",
        "",
        raw,
    ).lower()

    aliases = {
        "北海道コンサドーレ札幌": "札幌",
        "コンサドーレ札幌": "札幌",
        "札幌": "札幌",
        "ヴァンラーレ八戸": "八戸",
        "八戸": "八戸",
        "いわてグルージャ盛岡": "岩手",
        "グルージャ盛岡": "岩手",
        "岩手": "岩手",
        "ベガルタ仙台": "仙台",
        "仙台": "仙台",
        "ブラウブリッツ秋田": "秋田",
        "秋田": "秋田",
        "モンテディオ山形": "山形",
        "山形": "山形",
        "福島ユナイテッドfc": "福島",
        "福島ユナイテッド": "福島",
        "福島": "福島",
        "いわきfc": "いわき",
        "いわき": "いわき",
        "水戸ホーリーホック": "水戸",
        "水戸": "水戸",
        "栃木sc": "栃木sc",
        "栃木": "栃木sc",
        "栃木シティ": "栃木c",
        "栃木c": "栃木c",
        "ザスパ群馬": "群馬",
        "ザスパクサツ群馬": "群馬",
        "群馬": "群馬",
        "鹿島アントラーズ": "鹿島",
        "鹿島": "鹿島",
        "浦和レッズ": "浦和",
        "浦和": "浦和",
        "大宮アルディージャ": "大宮",
        "rb大宮アルディージャ": "大宮",
        "大宮": "大宮",
        "ジェフユナイテッド千葉": "千葉",
        "ジェフ千葉": "千葉",
        "千葉": "千葉",
        "柏レイソル": "柏",
        "柏": "柏",
        "fc東京": "fc東京",
        "fctokyo": "fc東京",
        "東京ヴェルディ": "東京v",
        "東京v": "東京v",
        "fc町田ゼルビア": "町田",
        "町田ゼルビア": "町田",
        "町田": "町田",
        "川崎フロンターレ": "川崎f",
        "川崎f": "川崎f",
        "横浜fマリノス": "横浜fm",
        "横浜fm": "横浜fm",
        "横浜fc": "横浜fc",
        "yscc横浜": "ys横浜",
        "ys横浜": "ys横浜",
        "湘南ベルマーレ": "湘南",
        "湘南": "湘南",
        "sc相模原": "相模原",
        "相模原": "相模原",
        "ヴァンフォーレ甲府": "甲府",
        "甲府": "甲府",
        "松本山雅fc": "松本",
        "松本山雅": "松本",
        "松本": "松本",
        "ac長野パルセイロ": "長野",
        "長野パルセイロ": "長野",
        "長野": "長野",
        "アルビレックス新潟": "新潟",
        "新潟": "新潟",
        "カターレ富山": "富山",
        "富山": "富山",
        "ツエーゲン金沢": "金沢",
        "金沢": "金沢",
        "清水エスパルス": "清水",
        "清水": "清水",
        "ジュビロ磐田": "磐田",
        "磐田": "磐田",
        "藤枝myfc": "藤枝",
        "藤枝": "藤枝",
        "アスルクラロ沼津": "沼津",
        "沼津": "沼津",
        "名古屋グランパス": "名古屋",
        "名古屋": "名古屋",
        "fc岐阜": "岐阜",
        "岐阜": "岐阜",
        "京都サンガfc": "京都",
        "京都サンガ": "京都",
        "京都": "京都",
        "ガンバ大阪": "g大阪",
        "g大阪": "g大阪",
        "セレッソ大阪": "c大阪",
        "c大阪": "c大阪",
        "fc大阪": "fc大阪",
        "大阪": "fc大阪",
        "ヴィッセル神戸": "神戸",
        "神戸": "神戸",
        "奈良クラブ": "奈良",
        "奈良": "奈良",
        "レイラック滋賀": "滋賀",
        "mioびわこ滋賀": "滋賀",
        "滋賀": "滋賀",
        "ガイナーレ鳥取": "鳥取",
        "鳥取": "鳥取",
        "ファジアーノ岡山": "岡山",
        "岡山": "岡山",
        "サンフレッチェ広島": "広島",
        "広島": "広島",
        "レノファ山口fc": "山口",
        "レノファ山口": "山口",
        "山口": "山口",
        "徳島ヴォルティス": "徳島",
        "徳島": "徳島",
        "愛媛fc": "愛媛",
        "愛媛": "愛媛",
        "fc今治": "今治",
        "今治": "今治",
        "カマタマーレ讃岐": "讃岐",
        "讃岐": "讃岐",
        "高知ユナイテッドsc": "高知",
        "高知ユナイテッド": "高知",
        "高知": "高知",
        "アビスパ福岡": "福岡",
        "福岡": "福岡",
        "ギラヴァンツ北九州": "北九州",
        "北九州": "北九州",
        "サガン鳥栖": "鳥栖",
        "鳥栖": "鳥栖",
        "vファーレン長崎": "長崎",
        "長崎": "長崎",
        "ロアッソ熊本": "熊本",
        "熊本": "熊本",
        "大分トリニータ": "大分",
        "大分": "大分",
        "テゲバジャーロ宮崎": "宮崎",
        "宮崎": "宮崎",
        "鹿児島ユナイテッドfc": "鹿児島",
        "鹿児島ユナイテッド": "鹿児島",
        "鹿児島": "鹿児島",
        "fc琉球": "琉球",
        "琉球": "琉球",
    }

    return aliases.get(compact, compact)


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
        frame["kickoff_at_raw"] = frame["kickoff_at"].astype(str)
        frame["kickoff_at"] = pd.to_datetime(
            frame["kickoff_at"],
            errors="coerce",
            format="mixed",
        )
        frame["actual"] = frame["toto_result"].map(normalize_outcome)
        frame["home_key"] = frame["home_team"].map(clean_text)
        frame["away_key"] = frame["away_team"].map(clean_text)

        return frame.loc[
            frame["round_no"].notna()
            & frame["match_no"].notna()
            & frame["actual"].isin(CLASS_ORDER)
        ].copy()


def resolve_pair_candidates(
    fixture_home: object,
    fixture_away: object,
    predictions: pd.DataFrame,
    prediction_lookup: dict[tuple[str, str], list[int]],
) -> list[int]:
    """Return candidate prediction rows for one normalized fixture pair."""
    key = (
        clean_text(fixture_home),
        clean_text(fixture_away),
    )
    return prediction_lookup.get(key, [])


def match_predictions_to_fixtures(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    date_window_days: int = 3,
    jleague_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Match historical toto fixtures to rolling out-of-sample v6 predictions.

    Historical backfill now stores full kickoff dates, so the primary key is:
        normalized home team + normalized away team + nearest full date.

    For J.League-only evaluation, fixtures whose team pair never appears in the
    v6 prediction universe are classified as out-of-scope and excluded from the
    coverage denominator. This prevents overseas toto matches from making a
    valid J.League round look incomplete.
    """
    prediction_lookup: dict[tuple[str, str], list[int]] = {}
    for index, row in predictions.iterrows():
        key = (str(row["home_key"]), str(row["away_key"]))
        prediction_lookup.setdefault(key, []).append(index)

    prediction_team_keys = (
        set(predictions["home_key"].astype(str))
        | set(predictions["away_key"].astype(str))
    )

    matched_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for round_no, round_fixtures in fixtures.groupby("round_no", sort=True):
        round_fixtures = round_fixtures.sort_values(
            "match_no",
            kind="stable",
        ).reset_index(drop=True)

        round_matched = 0
        in_scope_count = 0
        out_of_scope_count = 0
        no_pair_count = 0
        date_mismatch_count = 0

        valid_round_dates = round_fixtures["kickoff_at"].dropna()
        inferred_round_date = (
            pd.Timestamp(valid_round_dates.min()).strftime("%Y-%m-%d")
            if not valid_round_dates.empty
            else None
        )

        for fixture in round_fixtures.itertuples(index=False):
            home_key = clean_text(fixture.home_team)
            away_key = clean_text(fixture.away_team)

            teams_in_prediction_universe = (
                home_key in prediction_team_keys
                and away_key in prediction_team_keys
            )

            if jleague_only and not teams_in_prediction_universe:
                out_of_scope_count += 1
                unmatched_rows.append(
                    {
                        "round_no": int(fixture.round_no),
                        "match_no": int(fixture.match_no),
                        "home_team": fixture.home_team,
                        "away_team": fixture.away_team,
                        "reason": "out_of_v6_jleague_scope",
                        "kickoff_at": (
                            pd.Timestamp(fixture.kickoff_at).strftime("%Y-%m-%d")
                            if pd.notna(fixture.kickoff_at)
                            else None
                        ),
                    }
                )
                continue

            in_scope_count += 1
            candidates = resolve_pair_candidates(
                fixture.home_team,
                fixture.away_team,
                predictions,
                prediction_lookup,
            )

            if not candidates:
                no_pair_count += 1
                unmatched_rows.append(
                    {
                        "round_no": int(fixture.round_no),
                        "match_no": int(fixture.match_no),
                        "home_team": fixture.home_team,
                        "away_team": fixture.away_team,
                        "reason": "no_exact_pair_match",
                        "kickoff_at": (
                            pd.Timestamp(fixture.kickoff_at).strftime("%Y-%m-%d")
                            if pd.notna(fixture.kickoff_at)
                            else None
                        ),
                    }
                )
                continue

            candidate_frame = predictions.loc[candidates].copy()
            kickoff = fixture.kickoff_at

            if pd.isna(kickoff):
                date_mismatch_count += 1
                unmatched_rows.append(
                    {
                        "round_no": int(fixture.round_no),
                        "match_no": int(fixture.match_no),
                        "home_team": fixture.home_team,
                        "away_team": fixture.away_team,
                        "reason": "missing_full_kickoff_date",
                    }
                )
                continue

            kickoff_date = pd.Timestamp(kickoff).normalize()
            candidate_frame["date_distance"] = (
                candidate_frame["match_date"].dt.normalize()
                - kickoff_date
            ).abs().dt.days
            candidate_frame = candidate_frame.sort_values(
                ["date_distance", "match_date"],
                kind="stable",
            )
            best = candidate_frame.iloc[0]
            date_distance = int(best["date_distance"])

            if date_distance > date_window_days:
                date_mismatch_count += 1
                unmatched_rows.append(
                    {
                        "round_no": int(fixture.round_no),
                        "match_no": int(fixture.match_no),
                        "home_team": fixture.home_team,
                        "away_team": fixture.away_team,
                        "reason": "date_distance_gt_window",
                        "date_distance_days": date_distance,
                        "kickoff_at": kickoff_date.strftime("%Y-%m-%d"),
                        "closest_prediction_date": pd.Timestamp(
                            best["match_date"]
                        ).strftime("%Y-%m-%d"),
                    }
                )
                continue

            row = best.to_dict()
            row.update(
                {
                    "round_no": int(fixture.round_no),
                    "toto_match_no": int(fixture.match_no),
                    "fixture_home_team": fixture.home_team,
                    "fixture_away_team": fixture.away_team,
                    "fixture_actual": fixture.actual,
                    "inferred_round_date": inferred_round_date,
                    "date_distance_days": date_distance,
                }
            )
            matched_rows.append(row)
            round_matched += 1

        denominator = (
            in_scope_count
            if jleague_only
            else len(round_fixtures)
        )
        coverage_rows.append(
            {
                "round_no": int(round_no),
                "fixture_count": len(round_fixtures),
                "jleague_in_scope_count": in_scope_count,
                "out_of_scope_count": out_of_scope_count,
                "matched_count": round_matched,
                "coverage_rate": (
                    round_matched / denominator
                    if denominator
                    else 0.0
                ),
                "no_team_match_count": no_pair_count,
                "date_mismatch_count": date_mismatch_count,
                "inferred_round_date": inferred_round_date,
            }
        )

    return (
        pd.DataFrame(matched_rows),
        pd.DataFrame(unmatched_rows),
        pd.DataFrame(coverage_rows),
    )


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

    fixture_count = int(
        round_frame["jleague_in_scope_count"].iloc[0]
        if "jleague_in_scope_count" in round_frame.columns
        else (
            round_frame["fixture_count"].iloc[0]
            if "fixture_count" in round_frame.columns
            else match_count
        )
    )
    coverage_rate = (
        match_count / fixture_count if fixture_count else 0.0
    )

    summary = {
        "round_no": int(round_frame["round_no"].iloc[0]),
        "fixture_count": fixture_count,
        "match_count": match_count,
        "coverage_rate": coverage_rate,
        "inferred_round_date": (
            str(round_frame["inferred_round_date"].iloc[0])
            if "inferred_round_date" in round_frame.columns
            else None
        ),
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
    parser.add_argument("--date-window-days", type=int, default=3)
    parser.add_argument(
        "--minimum-round-coverage",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--all-fixtures",
        action="store_true",
        help=(
            "Evaluate all toto fixtures. Default is J.League-only, "
            "because the v6 model was trained on J.League matches."
        ),
    )
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
        date_window_days=args.date_window_days,
        minimum_round_coverage=args.minimum_round_coverage,
        jleague_only=not args.all_fixtures,
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
    matched, unmatched, coverage = match_predictions_to_fixtures(
        fixtures=fixtures,
        predictions=predictions,
        date_window_days=config.date_window_days,
        jleague_only=config.jleague_only,
    )

    if matched.empty:
        raise TrueBacktestError(
            "No historical toto fixtures matched v6 prediction rows."
        )

    eligible_rounds = set(
        coverage.loc[
            (coverage["jleague_in_scope_count"] > 0)
            & (
                coverage["coverage_rate"]
                >= config.minimum_round_coverage
            ),
            "round_no",
        ].astype(int)
    )
    excluded_rounds = coverage.loc[
        ~coverage["round_no"].astype(int).isin(eligible_rounds)
    ].copy()

    # 照合率不足で停止する場合でも、原因確認用CSVを必ず保存します。
    config.output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(
        config.output_dir / "v6_true_backtest_round_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unmatched.to_csv(
        config.output_dir / "v6_true_backtest_unmatched.csv",
        index=False,
        encoding="utf-8-sig",
    )
    excluded_rounds.to_csv(
        config.output_dir / "v6_true_backtest_excluded_rounds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    matched = matched.loc[
        matched["round_no"].astype(int).isin(eligible_rounds)
    ].copy()

    if matched.empty:
        raise TrueBacktestError(
            "No rounds met the minimum coverage threshold. "
            "Diagnostics were saved to v6_true_backtest_round_coverage.csv "
            "and v6_true_backtest_unmatched.csv."
        )

    matched = matched.merge(
        coverage[
            [
                "round_no",
                "fixture_count",
                "jleague_in_scope_count",
                "coverage_rate",
            ]
        ],
        on="round_no",
        how="left",
        validate="many_to_one",
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
    coverage.to_csv(
        config.output_dir / "v6_true_backtest_round_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    excluded_rounds.to_csv(
        config.output_dir / "v6_true_backtest_excluded_rounds.csv",
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
    print(f"Rounds in fixture cache    : {len(coverage)}")
    print(f"Rounds evaluated           : {overall['rounds_evaluated']}")
    eligible_coverage = coverage.loc[
        coverage["jleague_in_scope_count"] > 0,
        "coverage_rate",
    ]
    print(
        "Mean J.League coverage     : "
        f"{eligible_coverage.mean():.4f}"
        if not eligible_coverage.empty
        else "Mean J.League coverage     : 0.0000"
    )
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
