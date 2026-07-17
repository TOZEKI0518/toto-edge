from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger("training_dataset_generator_v2")


TEAM_ALIASES: dict[str, str] = {
    "北海道コンサドーレ札幌": "札幌",
    "コンサドーレ札幌": "札幌",
    "鹿島アントラーズ": "鹿島",
    "浦和レッズ": "浦和",
    "柏レイソル": "柏",
    "ＦＣ東京": "FC東京",
    "FC東京": "FC東京",
    "東京ヴェルディ": "東京V",
    "東京Ｖ": "東京V",
    "川崎フロンターレ": "川崎F",
    "川崎Ｆ": "川崎F",
    "横浜Ｆ・マリノス": "横浜FM",
    "横浜F・マリノス": "横浜FM",
    "横浜ＦＣ": "横浜FC",
    "湘南ベルマーレ": "湘南",
    "アルビレックス新潟": "新潟",
    "清水エスパルス": "清水",
    "名古屋グランパス": "名古屋",
    "京都サンガＦ．Ｃ．": "京都",
    "京都サンガF.C.": "京都",
    "ガンバ大阪": "G大阪",
    "Ｇ大阪": "G大阪",
    "セレッソ大阪": "C大阪",
    "Ｃ大阪": "C大阪",
    "ヴィッセル神戸": "神戸",
    "ファジアーノ岡山": "岡山",
    "サンフレッチェ広島": "広島",
    "アビスパ福岡": "福岡",
    "サガン鳥栖": "鳥栖",
    "ＦＣ町田ゼルビア": "町田",
    "FC町田ゼルビア": "町田",
    "ベガルタ仙台": "仙台",
    "ブラウブリッツ秋田": "秋田",
    "モンテディオ山形": "山形",
    "いわきＦＣ": "いわき",
    "水戸ホーリーホック": "水戸",
    "栃木ＳＣ": "栃木",
    "ザスパ群馬": "群馬",
    "ジェフユナイテッド千葉": "千葉",
    "ヴァンフォーレ甲府": "甲府",
    "カターレ富山": "富山",
    "ジュビロ磐田": "磐田",
    "藤枝ＭＹＦＣ": "藤枝",
    "レノファ山口ＦＣ": "山口",
    "徳島ヴォルティス": "徳島",
    "愛媛ＦＣ": "愛媛",
    "Ｖ・ファーレン長崎": "長崎",
    "ロアッソ熊本": "熊本",
    "大分トリニータ": "大分",
    "大宮アルディージャ": "大宮",
    "RB大宮アルディージャ": "大宮",
}


@dataclass(frozen=True)
class GeneratorPaths:
    matches_csv: Path
    team_features_csv: Path
    position_features_csv: Path
    output_csv: Path
    diagnostics_dir: Path


@dataclass(frozen=True)
class GenerationReport:
    rows: int
    columns: int
    numeric_features: int
    duplicate_match_ids: int
    missing_match_ids: int
    missing_dates: int
    missing_home_teams: int
    missing_away_teams: int
    missing_results: int
    team_feature_match_rows: int
    position_feature_match_rows: int


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_paths() -> GeneratorPaths:
    root = project_root()
    processed = root / "ml" / "player_engine" / "data" / "processed"
    return GeneratorPaths(
        matches_csv=processed / "matches.csv",
        team_features_csv=processed / "historical_team_features.csv",
        position_features_csv=processed / "historical_position_features.csv",
        output_csv=root / "ml" / "training_dataset_v2.csv",
        diagnostics_dir=root / "ml" / "diagnostics" / "training_dataset_v2",
    )


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{label} is empty: {path}")

    LOGGER.info("%s rows: %s", label, f"{len(frame):,}")
    return frame


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    label: str,
    required: bool = True,
) -> str | None:
    lower_map = {str(column).lower(): str(column) for column in frame.columns}

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        matched = lower_map.get(candidate.lower())
        if matched is not None:
            return matched

    if required:
        raise ValueError(
            f"{label} does not contain any of these columns: "
            f"{list(candidates)}. Available: {list(frame.columns)}"
        )
    return None


def normalize_team(value: object) -> str:
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("･", "・")
    return TEAM_ALIASES.get(text, text)


def normalize_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").fillna("")


def normalize_match_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def determine_result(home_score: object, away_score: object) -> str:
    if pd.isna(home_score) or pd.isna(away_score):
        return ""
    if float(home_score) > float(away_score):
        return "H"
    if float(home_score) < float(away_score):
        return "A"
    return "D"


def prepare_matches(matches: pd.DataFrame) -> pd.DataFrame:
    match_id_col = find_column(
        matches,
        ["match_card_id", "matchId", "match_id"],
        "matches.csv",
    )
    home_col = find_column(
        matches,
        ["home_team", "homeTeam", "home"],
        "matches.csv",
    )
    away_col = find_column(
        matches,
        ["away_team", "awayTeam", "away"],
        "matches.csv",
    )
    date_col = find_column(
        matches,
        ["match_date", "matchDate", "date"],
        "matches.csv",
    )

    season_col = find_column(
        matches,
        ["season", "competition_year", "year"],
        "matches.csv",
        required=False,
    )
    competition_col = find_column(
        matches,
        ["competition", "competition_name", "league", "category"],
        "matches.csv",
        required=False,
    )
    round_col = find_column(
        matches,
        ["round", "section", "match_round", "節"],
        "matches.csv",
        required=False,
    )
    home_score_col = find_column(
        matches,
        ["home_score", "homeScore", "score_home"],
        "matches.csv",
        required=False,
    )
    away_score_col = find_column(
        matches,
        ["away_score", "awayScore", "score_away"],
        "matches.csv",
        required=False,
    )
    result_col = find_column(
        matches,
        ["result", "actual", "target", "label"],
        "matches.csv",
        required=False,
    )

    result = pd.DataFrame(
        {
            "match_card_id": normalize_match_id(matches[match_id_col]),
            "match_date": normalize_date(matches[date_col]),
            "home_team": matches[home_col].map(normalize_team),
            "away_team": matches[away_col].map(normalize_team),
        }
    )

    if season_col is not None:
        result["season"] = safe_numeric(matches[season_col]).astype("Int64")
    else:
        result["season"] = pd.to_numeric(
            result["match_date"].str[:4],
            errors="coerce",
        ).astype("Int64")

    result["competition"] = (
        matches[competition_col].astype(str).fillna("")
        if competition_col is not None
        else ""
    )
    result["round"] = (
        safe_numeric(matches[round_col]).astype("Int64")
        if round_col is not None
        else pd.Series(pd.NA, index=matches.index, dtype="Int64")
    )

    result["home_score"] = (
        safe_numeric(matches[home_score_col])
        if home_score_col is not None
        else pd.Series(float("nan"), index=matches.index)
    )
    result["away_score"] = (
        safe_numeric(matches[away_score_col])
        if away_score_col is not None
        else pd.Series(float("nan"), index=matches.index)
    )

    if result_col is not None:
        normalized_result = (
            matches[result_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"1": "H", "0": "D", "2": "A"})
        )
        result["result"] = normalized_result.where(
            normalized_result.isin(["H", "D", "A"]),
            "",
        )
    else:
        result["result"] = [
            determine_result(home_score, away_score)
            for home_score, away_score in zip(
                result["home_score"],
                result["away_score"],
                strict=False,
            )
        ]

    duplicate_count = int(result["match_card_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"matches.csv has duplicate match_card_id rows: {duplicate_count}"
        )

    return result


def prepare_feature_table(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    match_id_col = find_column(
        frame,
        ["match_card_id", "matchId", "match_id"],
        label,
    )
    team_col = find_column(
        frame,
        ["team", "team_name"],
        label,
    )

    result = frame.copy()
    result["match_card_id"] = normalize_match_id(result[match_id_col])
    result["_team_key"] = result[team_col].map(normalize_team)

    result = result.dropna(subset=["match_card_id"])

    duplicates = int(
        result.duplicated(["match_card_id", "_team_key"]).sum()
    )
    if duplicates:
        raise ValueError(
            f"{label} has duplicate match/team keys: {duplicates}"
        )

    return result


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "match_card_id",
        "matchId",
        "match_id",
        "team",
        "team_name",
        "_team_key",
        "season",
        "round",
        "match_date",
        "date",
        "competition",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def merge_features(
    training: pd.DataFrame,
    features: pd.DataFrame,
    columns: list[str],
    side: str,
) -> pd.DataFrame:
    side_team = f"{side}_team"
    rename_map = {
        column: f"{side}_{column}"
        for column in columns
    }

    lookup = features[
        ["match_card_id", "_team_key", *columns]
    ].rename(columns=rename_map)

    merged = training.merge(
        lookup,
        left_on=["match_card_id", side_team],
        right_on=["match_card_id", "_team_key"],
        how="left",
        validate="one_to_one",
    )

    return merged.drop(columns=["_team_key"])


def add_pair_features(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    additions: dict[str, pd.Series] = {}

    for column in columns:
        home = f"home_{column}"
        away = f"away_{column}"

        if home not in frame.columns or away not in frame.columns:
            continue

        additions[f"{column}Diff"] = frame[home] - frame[away]

        denominator = frame[away].abs().where(
            frame[away].abs() > 1e-9
        )
        additions[f"{column}Ratio"] = frame[home] / denominator

    if additions:
        frame = pd.concat(
            [frame, pd.DataFrame(additions, index=frame.index)],
            axis=1,
        )

    return frame


def build_feature_registry(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "match_card_id",
        "season",
        "round",
        "home_score",
        "away_score",
        "result",
    }
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in excluded
    ]


def build_dataset(
    matches: pd.DataFrame,
    team_features: pd.DataFrame,
    position_features: pd.DataFrame,
) -> tuple[pd.DataFrame, GenerationReport]:
    training = prepare_matches(matches)
    prepared_team = prepare_feature_table(
        team_features,
        "historical_team_features.csv",
    )
    prepared_position = prepare_feature_table(
        position_features,
        "historical_position_features.csv",
    )

    team_columns = numeric_feature_columns(prepared_team)
    position_columns = numeric_feature_columns(prepared_position)

    training = merge_features(
        training,
        prepared_team,
        team_columns,
        "home",
    )
    training = merge_features(
        training,
        prepared_team,
        team_columns,
        "away",
    )
    training = merge_features(
        training,
        prepared_position,
        position_columns,
        "home",
    )
    training = merge_features(
        training,
        prepared_position,
        position_columns,
        "away",
    )

    training = add_pair_features(
        training,
        [*team_columns, *position_columns],
    )

    home_team_feature_cols = [
        f"home_{column}" for column in team_columns
    ]
    away_team_feature_cols = [
        f"away_{column}" for column in team_columns
    ]
    home_position_feature_cols = [
        f"home_{column}" for column in position_columns
    ]
    away_position_feature_cols = [
        f"away_{column}" for column in position_columns
    ]

    team_matched = (
        training[home_team_feature_cols].notna().any(axis=1)
        & training[away_team_feature_cols].notna().any(axis=1)
        if home_team_feature_cols and away_team_feature_cols
        else pd.Series(False, index=training.index)
    )
    position_matched = (
        training[home_position_feature_cols].notna().any(axis=1)
        & training[away_position_feature_cols].notna().any(axis=1)
        if home_position_feature_cols and away_position_feature_cols
        else pd.Series(False, index=training.index)
    )

    training["team_feature_matched"] = team_matched.astype(int)
    training["position_feature_matched"] = position_matched.astype(int)

    registry = build_feature_registry(training)

    report = GenerationReport(
        rows=len(training),
        columns=len(training.columns),
        numeric_features=len(registry),
        duplicate_match_ids=int(
            training["match_card_id"].duplicated().sum()
        ),
        missing_match_ids=int(training["match_card_id"].isna().sum()),
        missing_dates=int(training["match_date"].eq("").sum()),
        missing_home_teams=int(training["home_team"].eq("").sum()),
        missing_away_teams=int(training["away_team"].eq("").sum()),
        missing_results=int(training["result"].eq("").sum()),
        team_feature_match_rows=int(team_matched.sum()),
        position_feature_match_rows=int(position_matched.sum()),
    )

    return training, report


def save_outputs(
    training: pd.DataFrame,
    report: GenerationReport,
    paths: GeneratorPaths,
) -> None:
    paths.output_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    training.to_csv(
        paths.output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    registry = build_feature_registry(training)

    (paths.diagnostics_dir / "feature_columns.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "feature": registry,
            "dtype": [str(training[column].dtype) for column in registry],
            "missing": [
                int(training[column].isna().sum())
                for column in registry
            ],
        }
    ).to_csv(
        paths.diagnostics_dir / "feature_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "column": training.columns,
            "missing": [
                int(training[column].isna().sum())
                for column in training.columns
            ],
        }
    ).sort_values(
        "missing",
        ascending=False,
    ).to_csv(
        paths.diagnostics_dir / "missing_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        **asdict(report),
        "created_at": datetime.now().isoformat(),
        "output_csv": str(paths.output_csv),
    }
    (paths.diagnostics_dir / "generation_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(
        description="Project Alpha Training Dataset Generator v2."
    )
    parser.add_argument(
        "--matches",
        type=Path,
        default=defaults.matches_csv,
    )
    parser.add_argument(
        "--team-features",
        type=Path,
        default=defaults.team_features_csv,
    )
    parser.add_argument(
        "--position-features",
        type=Path,
        default=defaults.position_features_csv,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults.output_csv,
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=defaults.diagnostics_dir,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    paths = GeneratorPaths(
        matches_csv=args.matches,
        team_features_csv=args.team_features,
        position_features_csv=args.position_features,
        output_csv=args.output,
        diagnostics_dir=args.diagnostics_dir,
    )

    matches = load_csv(paths.matches_csv, "Matches")
    team_features = load_csv(
        paths.team_features_csv,
        "Historical team features",
    )
    position_features = load_csv(
        paths.position_features_csv,
        "Historical position features",
    )

    training, report = build_dataset(
        matches,
        team_features,
        position_features,
    )
    save_outputs(training, report, paths)

    print("=" * 72)
    print("Project Alpha Training Dataset Generator v2")
    print("=" * 72)
    print(f"Rows                        : {report.rows}")
    print(f"Columns                     : {report.columns}")
    print(f"Numeric features            : {report.numeric_features}")
    print(f"Duplicate match IDs         : {report.duplicate_match_ids}")
    print(f"Missing match IDs           : {report.missing_match_ids}")
    print(f"Missing dates               : {report.missing_dates}")
    print(f"Missing home teams          : {report.missing_home_teams}")
    print(f"Missing away teams          : {report.missing_away_teams}")
    print(f"Missing results             : {report.missing_results}")
    print(f"Team feature matched rows   : {report.team_feature_match_rows}")
    print(
        f"Position feature matched rows: "
        f"{report.position_feature_match_rows}"
    )
    print(f"Saved                       : {paths.output_csv}")
    print(f"Diagnostics                 : {paths.diagnostics_dir}")


if __name__ == "__main__":
    main()
