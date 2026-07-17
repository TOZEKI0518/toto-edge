from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger("matching_engine_v2")

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
class EnginePaths:
    training_csv: Path
    team_features_csv: Path
    position_features_csv: Path
    output_csv: Path
    diagnostics_dir: Path


@dataclass(frozen=True)
class MatchingReport:
    training_rows: int
    matched_rows: int
    unmatched_rows: int
    match_rate: float
    team_feature_columns: int
    position_feature_columns: int
    output_columns: int


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_paths() -> EnginePaths:
    root = project_root()
    processed = root / "ml" / "player_engine" / "data" / "processed"
    return EnginePaths(
        training_csv=root / "ml" / "training_dataset_v2.csv",
        team_features_csv=processed / "historical_team_features.csv",
        position_features_csv=processed / "historical_position_features.csv",
        output_csv=root / "ml" / "training_dataset_v3_player.csv",
        diagnostics_dir=root / "ml" / "diagnostics" / "matching_v2",
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
) -> str:
    lower_map = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        matched = lower_map.get(candidate.lower())
        if matched is not None:
            return matched
    raise ValueError(f"{label} missing one of columns: {list(candidates)}")


def normalize_team(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("･", "・")
    return TEAM_ALIASES.get(text, text)


def normalize_match_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def prepare_training(frame: pd.DataFrame) -> pd.DataFrame:
    match_id_col = find_column(
        frame,
        ["match_card_id", "matchId", "match_id"],
        "training dataset",
    )
    home_col = find_column(
        frame,
        ["home_team", "homeTeam", "home"],
        "training dataset",
    )
    away_col = find_column(
        frame,
        ["away_team", "awayTeam", "away"],
        "training dataset",
    )

    result = frame.copy()
    result["match_card_id"] = normalize_match_id(result[match_id_col])
    result["_home_key"] = result[home_col].map(normalize_team)
    result["_away_key"] = result[away_col].map(normalize_team)

    if result["match_card_id"].isna().any():
        LOGGER.warning(
            "Training rows with missing match_card_id: %s",
            int(result["match_card_id"].isna().sum()),
        )

    return result


def prepare_feature_table(
    frame: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
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
    duplicate_count = int(
        result.duplicated(["match_card_id", "_team_key"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            f"{label} has duplicate match/team keys: {duplicate_count}"
        )

    return result


def feature_columns(frame: pd.DataFrame) -> list[str]:
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


def side_table(
    frame: pd.DataFrame,
    columns: list[str],
    side: str,
) -> pd.DataFrame:
    rename_map = {column: f"{side}_{column}" for column in columns}
    return frame[
        ["match_card_id", "_team_key", *columns]
    ].rename(columns=rename_map)


def merge_side(
    training: pd.DataFrame,
    features: pd.DataFrame,
    columns: list[str],
    side: str,
) -> pd.DataFrame:
    side_key = f"_{side}_key"
    prepared = side_table(features, columns, side)

    merged = training.merge(
        prepared,
        left_on=["match_card_id", side_key],
        right_on=["match_card_id", "_team_key"],
        how="left",
        validate="many_to_one",
    )
    return merged.drop(columns=["_team_key"])


def add_pair_features(
    frame: pd.DataFrame,
    base_columns: Iterable[str],
) -> pd.DataFrame:
    additions: dict[str, pd.Series] = {}

    for column in base_columns:
        home = f"home_{column}"
        away = f"away_{column}"
        if home not in frame.columns or away not in frame.columns:
            continue

        additions[f"{column}Diff"] = frame[home] - frame[away]

        denominator = frame[away].abs()
        safe_denominator = denominator.where(denominator > 1e-9)
        additions[f"{column}Ratio"] = frame[home] / safe_denominator

    if additions:
        frame = pd.concat(
            [frame, pd.DataFrame(additions, index=frame.index)],
            axis=1,
        )

    return frame


def build_model_feature_registry(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "actual",
        "result",
        "target",
        "label",
        "prediction",
        "match_card_id",
        "matchId",
        "match_id",
        "round",
        "roundNo",
        "season",
        "home_score",
        "away_score",
        "matching_method",
    }
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in excluded
    ]


def run_engine(
    training: pd.DataFrame,
    team_features: pd.DataFrame,
    position_features: pd.DataFrame,
) -> tuple[pd.DataFrame, MatchingReport]:
    prepared_training = prepare_training(training)
    prepared_team = prepare_feature_table(
        team_features,
        "historical_team_features",
    )
    prepared_position = prepare_feature_table(
        position_features,
        "historical_position_features",
    )

    team_columns = feature_columns(prepared_team)
    position_columns = feature_columns(prepared_position)

    merged = merge_side(
        prepared_training,
        prepared_team,
        team_columns,
        "home",
    )
    merged = merge_side(
        merged,
        prepared_team,
        team_columns,
        "away",
    )
    merged = merge_side(
        merged,
        prepared_position,
        position_columns,
        "home",
    )
    merged = merge_side(
        merged,
        prepared_position,
        position_columns,
        "away",
    )

    merged = add_pair_features(
        merged,
        [*team_columns, *position_columns],
    )

    home_probe_columns = [
        f"home_{column}"
        for column in [*team_columns, *position_columns]
        if f"home_{column}" in merged.columns
    ]
    away_probe_columns = [
        f"away_{column}"
        for column in [*team_columns, *position_columns]
        if f"away_{column}" in merged.columns
    ]

    if home_probe_columns and away_probe_columns:
        home_matched = merged[home_probe_columns].notna().any(axis=1)
        away_matched = merged[away_probe_columns].notna().any(axis=1)
        matched_mask = home_matched & away_matched
    else:
        matched_mask = pd.Series(False, index=merged.index)

    merged["matching_method"] = matched_mask.map(
        {True: "match_card_id_team", False: "unmatched"}
    )
    merged["matching_confidence"] = matched_mask.astype(float)

    matched_rows = int(matched_mask.sum())
    total_rows = len(merged)

    report = MatchingReport(
        training_rows=total_rows,
        matched_rows=matched_rows,
        unmatched_rows=total_rows - matched_rows,
        match_rate=matched_rows / total_rows if total_rows else 0.0,
        team_feature_columns=len(team_columns),
        position_feature_columns=len(position_columns),
        output_columns=len(merged.columns),
    )

    return merged.drop(columns=["_home_key", "_away_key"]), report


def save_outputs(
    frame: pd.DataFrame,
    report: MatchingReport,
    paths: EnginePaths,
) -> None:
    paths.output_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(
        paths.output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    unmatched = frame.loc[
        frame["matching_method"].eq("unmatched")
    ].copy()
    unmatched.to_csv(
        paths.diagnostics_dir / "matching_unmatched.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "matching_method": method,
                "rows": int(rows),
            }
            for method, rows in frame["matching_method"]
            .value_counts()
            .items()
        ]
    ).to_csv(
        paths.diagnostics_dir / "matching_method_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    registry = build_model_feature_registry(frame)
    (paths.diagnostics_dir / "feature_columns.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "feature": registry,
            "dtype": [str(frame[column].dtype) for column in registry],
            "missing": [
                int(frame[column].isna().sum()) for column in registry
            ],
        }
    ).to_csv(
        paths.diagnostics_dir / "feature_registry_v3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (paths.diagnostics_dir / "matching_summary.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(
        description="Project Alpha Matching Engine v2."
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=defaults.training_csv,
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

    paths = EnginePaths(
        training_csv=args.training,
        team_features_csv=args.team_features,
        position_features_csv=args.position_features,
        output_csv=args.output,
        diagnostics_dir=args.diagnostics_dir,
    )

    training = load_csv(paths.training_csv, "Training dataset")
    team_features = load_csv(
        paths.team_features_csv,
        "Historical team features",
    )
    position_features = load_csv(
        paths.position_features_csv,
        "Historical position features",
    )

    merged, report = run_engine(
        training,
        team_features,
        position_features,
    )
    save_outputs(merged, report, paths)

    print("=" * 72)
    print("Project Alpha Matching Engine v2")
    print("=" * 72)
    print(f"Training rows            : {report.training_rows}")
    print(f"Matched rows             : {report.matched_rows}")
    print(f"Unmatched rows           : {report.unmatched_rows}")
    print(f"Match rate               : {report.match_rate:.2%}")
    print(f"Team feature columns     : {report.team_feature_columns}")
    print(f"Position feature columns : {report.position_feature_columns}")
    print(f"Output columns           : {report.output_columns}")
    print(f"Saved                    : {paths.output_csv}")
    print(f"Diagnostics              : {paths.diagnostics_dir}")


if __name__ == "__main__":
    main()
