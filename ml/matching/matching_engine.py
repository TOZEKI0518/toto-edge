from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Iterable

import pandas as pd

LOGGER = logging.getLogger("matching_engine")

TEAM_ALIASES: Final[dict[str, str]] = {
    "北海道コンサドーレ札幌": "札幌", "コンサドーレ札幌": "札幌",
    "鹿島アントラーズ": "鹿島", "浦和レッズ": "浦和", "柏レイソル": "柏",
    "ＦＣ東京": "FC東京", "FC東京": "FC東京", "東京ヴェルディ": "東京V",
    "東京Ｖ": "東京V", "川崎フロンターレ": "川崎F", "川崎Ｆ": "川崎F",
    "横浜Ｆ・マリノス": "横浜FM", "横浜F・マリノス": "横浜FM",
    "横浜ＦＣ": "横浜FC", "横浜FC": "横浜FC", "湘南ベルマーレ": "湘南",
    "アルビレックス新潟": "新潟", "清水エスパルス": "清水",
    "名古屋グランパス": "名古屋", "京都サンガF.C.": "京都",
    "京都サンガＦ．Ｃ．": "京都", "ガンバ大阪": "G大阪", "Ｇ大阪": "G大阪",
    "セレッソ大阪": "C大阪", "Ｃ大阪": "C大阪", "ヴィッセル神戸": "神戸",
    "ファジアーノ岡山": "岡山", "サンフレッチェ広島": "広島",
    "アビスパ福岡": "福岡", "サガン鳥栖": "鳥栖",
    "ＦＣ町田ゼルビア": "町田", "FC町田ゼルビア": "町田", "町田ゼルビア": "町田",
    "ベガルタ仙台": "仙台", "ブラウブリッツ秋田": "秋田",
    "モンテディオ山形": "山形", "いわきＦＣ": "いわき", "いわきFC": "いわき",
    "水戸ホーリーホック": "水戸", "栃木ＳＣ": "栃木", "栃木SC": "栃木",
    "ザスパ群馬": "群馬", "ジェフユナイテッド千葉": "千葉",
    "ヴァンフォーレ甲府": "甲府", "カターレ富山": "富山",
    "ジュビロ磐田": "磐田", "藤枝ＭＹＦＣ": "藤枝", "藤枝MYFC": "藤枝",
    "レノファ山口ＦＣ": "山口", "レノファ山口FC": "山口",
    "徳島ヴォルティス": "徳島", "愛媛ＦＣ": "愛媛", "愛媛FC": "愛媛",
    "Ｖ・ファーレン長崎": "長崎", "V・ファーレン長崎": "長崎",
    "ロアッソ熊本": "熊本", "大分トリニータ": "大分",
    "RB大宮アルディージャ": "大宮", "大宮アルディージャ": "大宮",
}

# Adjust these ranges when new toto rounds are added.
DEFAULT_ROUND_TO_SEASON: Final[list[tuple[int, int, int]]] = [
    (1500, 1549, 2024),
    (1550, 9999, 2025),
]

CONFIDENCE: Final[dict[str, int]] = {
    "match_id": 100,
    "season_team_date": 95,
    "season_team": 90,
    "team_date": 80,
    "unique_team_pair": 60,
    "unmatched": 0,
}


@dataclass(frozen=True)
class MatchingConfig:
    min_confidence: int = 60
    allow_unique_team_pair: bool = True
    infer_season_from_round: bool = True


@dataclass(frozen=True)
class MatchingSummary:
    training_rows: int
    matched_rows: int
    unmatched_rows: int
    ambiguous_rows: int
    match_rate: float
    methods: dict[str, int]


class MatchingEngine:
    """Match training rows to official matches without accepting ambiguity."""

    def __init__(
        self,
        matches: pd.DataFrame,
        config: MatchingConfig | None = None,
        team_aliases: dict[str, str] | None = None,
        round_to_season: list[tuple[int, int, int]] | None = None,
    ) -> None:
        self.config = config or MatchingConfig()
        self.team_aliases = dict(TEAM_ALIASES)
        if team_aliases:
            self.team_aliases.update(team_aliases)
        self.round_to_season = round_to_season or list(DEFAULT_ROUND_TO_SEASON)
        self.matches = self._prepare_matches(matches)
        self._build_indexes()

    @staticmethod
    def _find_column(
        frame: pd.DataFrame,
        candidates: Iterable[str],
        label: str,
        required: bool = True,
    ) -> str | None:
        lower_map = {str(c).lower(): str(c) for c in frame.columns}
        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
            matched = lower_map.get(candidate.lower())
            if matched is not None:
                return matched
        if required:
            raise ValueError(f"{label} missing columns: {list(candidates)}")
        return None

    def normalize_team(self, value: object) -> str:
        if pd.isna(value):
            return ""
        text = unicodedata.normalize("NFKC", str(value)).strip()
        text = re.sub(r"\s+", "", text).replace("･", "・")
        return self.team_aliases.get(text, text)

    @staticmethod
    def normalize_date(value: object) -> str:
        parsed = pd.to_datetime(value, errors="coerce")
        return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")

    @staticmethod
    def normalize_season(value: object) -> int | None:
        if pd.isna(value):
            return None
        match = re.search(r"(20\d{2})", str(value))
        return int(match.group(1)) if match else None

    def infer_season(self, round_value: object) -> int | None:
        numeric = pd.to_numeric(pd.Series([round_value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return None
        round_no = int(numeric)
        for start, end, season in self.round_to_season:
            if start <= round_no <= end:
                return season
        return None

    def _prepare_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        id_col = self._find_column(matches, ["match_card_id", "matchId", "match_id"], "matches")
        home_col = self._find_column(matches, ["home_team", "homeTeam", "home"], "matches")
        away_col = self._find_column(matches, ["away_team", "awayTeam", "away"], "matches")
        date_col = self._find_column(matches, ["match_date", "matchDate", "date"], "matches", False)
        season_col = self._find_column(matches, ["season", "competition_year", "year"], "matches", False)

        result = pd.DataFrame({
            "match_card_id": pd.to_numeric(matches[id_col], errors="coerce").astype("Int64"),
            "home_team": matches[home_col].map(self.normalize_team),
            "away_team": matches[away_col].map(self.normalize_team),
        })
        result["match_date"] = matches[date_col].map(self.normalize_date) if date_col else ""
        if season_col:
            result["season"] = matches[season_col].map(self.normalize_season)
        else:
            result["season"] = result["match_date"].map(lambda x: int(x[:4]) if x else None)

        result = result.dropna(subset=["match_card_id"]).copy()
        result["match_card_id"] = result["match_card_id"].astype(int)
        return result.drop_duplicates("match_card_id", keep="last").reset_index(drop=True)

    def _build_indexes(self) -> None:
        self.by_id: dict[int, list[int]] = defaultdict(list)
        self.by_season_team_date: dict[tuple[int, str, str, str], list[int]] = defaultdict(list)
        self.by_season_team: dict[tuple[int, str, str], list[int]] = defaultdict(list)
        self.by_team_date: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        self.by_team_pair: dict[tuple[str, str], list[int]] = defaultdict(list)

        for row in self.matches.itertuples(index=False):
            match_id = int(row.match_card_id)
            season = int(row.season) if row.season is not None else None
            home, away, date = str(row.home_team), str(row.away_team), str(row.match_date)
            self.by_id[match_id].append(match_id)
            self.by_team_pair[(home, away)].append(match_id)
            if date:
                self.by_team_date[(home, away, date)].append(match_id)
            if season is not None:
                self.by_season_team[(season, home, away)].append(match_id)
                if date:
                    self.by_season_team_date[(season, home, away, date)].append(match_id)

    @staticmethod
    def _resolve(candidates: list[int]) -> tuple[str, int | None, int]:
        unique = sorted(set(candidates))
        if len(unique) == 1:
            return "matched", unique[0], 1
        if len(unique) > 1:
            return "ambiguous", None, len(unique)
        return "unmatched", None, 0

    def _prepare_training(self, training: pd.DataFrame) -> pd.DataFrame:
        home_col = self._find_column(training, ["homeTeam", "home_team", "home"], "training")
        away_col = self._find_column(training, ["awayTeam", "away_team", "away"], "training")
        id_col = self._find_column(training, ["match_card_id", "matchId", "match_id"], "training", False)
        date_col = self._find_column(training, ["matchDate", "match_date", "date"], "training", False)
        season_col = self._find_column(training, ["season", "competition_year", "year"], "training", False)
        round_col = self._find_column(training, ["round", "roundNo", "totoRound", "toto_round"], "training", False)

        result = training.copy()
        result["_row_id"] = range(len(result))
        result["_home"] = result[home_col].map(self.normalize_team)
        result["_away"] = result[away_col].map(self.normalize_team)
        result["_date"] = result[date_col].map(self.normalize_date) if date_col else ""
        if season_col:
            result["_season"] = result[season_col].map(self.normalize_season)
        elif self.config.infer_season_from_round and round_col:
            result["_season"] = result[round_col].map(self.infer_season)
        else:
            result["_season"] = None
        result["_match_id"] = (
            pd.to_numeric(result[id_col], errors="coerce").astype("Int64")
            if id_col else pd.Series(pd.NA, index=result.index, dtype="Int64")
        )
        return result

    def _match_one(
        self,
        match_id: int | None,
        season: int | None,
        home: str,
        away: str,
        date: str,
    ) -> tuple[str, str, int, int | None, int]:
        attempts: list[tuple[str, list[int]]] = []
        if match_id is not None:
            attempts.append(("match_id", self.by_id.get(match_id, [])))
        if season is not None and date:
            attempts.append(("season_team_date", self.by_season_team_date.get((season, home, away, date), [])))
        if season is not None:
            attempts.append(("season_team", self.by_season_team.get((season, home, away), [])))
        if date:
            attempts.append(("team_date", self.by_team_date.get((home, away, date), [])))
        if self.config.allow_unique_team_pair:
            attempts.append(("unique_team_pair", self.by_team_pair.get((home, away), [])))

        first_ambiguous: tuple[str, int] | None = None
        for method, candidates in attempts:
            status, resolved, count = self._resolve(candidates)
            if status == "matched":
                confidence = CONFIDENCE[method]
                if confidence >= self.config.min_confidence:
                    return status, method, confidence, resolved, count
            elif status == "ambiguous" and first_ambiguous is None:
                first_ambiguous = (method, count)

        if first_ambiguous:
            return "ambiguous", first_ambiguous[0], 0, None, first_ambiguous[1]
        return "unmatched", "unmatched", 0, None, 0

    def match(self, training: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepare_training(training)
        rows: list[dict[str, object]] = []

        for row in prepared.itertuples(index=False):
            match_id_raw = row._match_id
            season_raw = row._season
            match_id = int(match_id_raw) if pd.notna(match_id_raw) else None
            season = int(season_raw) if season_raw is not None and pd.notna(season_raw) else None
            status, method, confidence, resolved, candidate_count = self._match_one(
                match_id, season, str(row._home), str(row._away), str(row._date)
            )
            rows.append({
                "_row_id": int(row._row_id),
                "matching_status": status,
                "matching_method": method,
                "matching_confidence": confidence,
                "matched_match_card_id": resolved,
                "matching_candidate_count": candidate_count,
                "matching_season": season,
                "matching_home_team": str(row._home),
                "matching_away_team": str(row._away),
                "matching_date": str(row._date),
            })

        decisions = pd.DataFrame(rows)
        result = prepared.merge(decisions, on="_row_id", how="left", validate="one_to_one")
        return result.drop(columns=["_row_id", "_home", "_away", "_date", "_season", "_match_id"])

    @staticmethod
    def summarize(result: pd.DataFrame) -> MatchingSummary:
        total = len(result)
        matched = int(result["matching_status"].eq("matched").sum())
        ambiguous = int(result["matching_status"].eq("ambiguous").sum())
        methods = dict(Counter(result["matching_method"].astype(str)))
        return MatchingSummary(
            training_rows=total,
            matched_rows=matched,
            unmatched_rows=total - matched,
            ambiguous_rows=ambiguous,
            match_rate=matched / total if total else 0.0,
            methods={str(k): int(v) for k, v in methods.items()},
        )

    @staticmethod
    def save_reports(result: pd.DataFrame, output_dir: Path) -> MatchingSummary:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = MatchingEngine.summarize(result)
        result.to_csv(output_dir / "matching_results.csv", index=False, encoding="utf-8-sig")
        result.loc[result["matching_status"].eq("ambiguous")].to_csv(
            output_dir / "matching_ambiguous.csv", index=False, encoding="utf-8-sig"
        )
        result.loc[result["matching_status"].eq("unmatched")].to_csv(
            output_dir / "matching_unmatched.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([
            {"method": method, "rows": rows}
            for method, rows in sorted(summary.methods.items(), key=lambda item: (-item[1], item[0]))
        ]).to_csv(output_dir / "matching_method_summary.csv", index=False, encoding="utf-8-sig")
        (output_dir / "matching_summary.json").write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = project_root()
    processed = root / "ml" / "player_engine" / "data" / "processed"
    parser = argparse.ArgumentParser(description="Project Alpha Matching Engine v2")
    parser.add_argument("--training", type=Path, default=root / "toto-training-dataset.csv")
    parser.add_argument("--matches", type=Path, default=processed / "matches.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "ml" / "diagnostics" / "matching_v2")
    parser.add_argument("--min-confidence", type=int, default=60)
    parser.add_argument("--disable-unique-team-pair", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if not args.training.exists():
        raise FileNotFoundError(f"Training dataset not found: {args.training}")
    if not args.matches.exists():
        raise FileNotFoundError(f"Matches dataset not found: {args.matches}")

    engine = MatchingEngine(
        pd.read_csv(args.matches),
        MatchingConfig(
            min_confidence=args.min_confidence,
            allow_unique_team_pair=not args.disable_unique_team_pair,
        ),
    )
    result = engine.match(pd.read_csv(args.training))
    summary = engine.save_reports(result, args.output_dir)

    print("=" * 72)
    print("Project Alpha Matching Engine v2")
    print("=" * 72)
    print(f"Training rows   : {summary.training_rows}")
    print(f"Matched rows    : {summary.matched_rows}")
    print(f"Unmatched rows  : {summary.unmatched_rows}")
    print(f"Ambiguous rows  : {summary.ambiguous_rows}")
    print(f"Match rate      : {summary.match_rate:.2%}")
    print("\nMatching methods")
    print("-" * 72)
    for method, rows in sorted(summary.methods.items(), key=lambda item: (-item[1], item[0])):
        print(f"{method:<28}: {rows}")
    print(f"\nSaved to        : {args.output_dir}")


if __name__ == "__main__":
    main()
