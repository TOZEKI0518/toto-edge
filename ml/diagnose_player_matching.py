from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


TEAM_ALIASES: dict[str, str] = {
    "北海道コンサドーレ札幌": "札幌",
    "コンサドーレ札幌": "札幌",
    "鹿島アントラーズ": "鹿島",
    "浦和レッズ": "浦和",
    "柏レイソル": "柏",
    "FC東京": "FC東京",
    "ＦＣ東京": "FC東京",
    "東京ヴェルディ": "東京V",
    "東京Ｖ": "東京V",
    "川崎フロンターレ": "川崎F",
    "川崎Ｆ": "川崎F",
    "横浜F・マリノス": "横浜FM",
    "横浜Ｆ・マリノス": "横浜FM",
    "横浜ＦＣ": "横浜FC",
    "湘南ベルマーレ": "湘南",
    "アルビレックス新潟": "新潟",
    "清水エスパルス": "清水",
    "名古屋グランパス": "名古屋",
    "京都サンガF.C.": "京都",
    "京都サンガＦ．Ｃ．": "京都",
    "ガンバ大阪": "G大阪",
    "Ｇ大阪": "G大阪",
    "セレッソ大阪": "C大阪",
    "Ｃ大阪": "C大阪",
    "ヴィッセル神戸": "神戸",
    "ファジアーノ岡山": "岡山",
    "サンフレッチェ広島": "広島",
    "アビスパ福岡": "福岡",
    "サガン鳥栖": "鳥栖",
    "FC町田ゼルビア": "町田",
    "ＦＣ町田ゼルビア": "町田",
    "町田ゼルビア": "町田",
    "ベガルタ仙台": "仙台",
    "ブラウブリッツ秋田": "秋田",
    "モンテディオ山形": "山形",
    "いわきFC": "いわき",
    "いわきＦＣ": "いわき",
    "水戸ホーリーホック": "水戸",
    "栃木SC": "栃木",
    "栃木ＳＣ": "栃木",
    "ザスパ群馬": "群馬",
    "ジェフユナイテッド千葉": "千葉",
    "ヴァンフォーレ甲府": "甲府",
    "カターレ富山": "富山",
    "ジュビロ磐田": "磐田",
    "藤枝MYFC": "藤枝",
    "藤枝ＭＹＦＣ": "藤枝",
    "レノファ山口FC": "山口",
    "レノファ山口ＦＣ": "山口",
    "徳島ヴォルティス": "徳島",
    "愛媛FC": "愛媛",
    "愛媛ＦＣ": "愛媛",
    "V・ファーレン長崎": "長崎",
    "Ｖ・ファーレン長崎": "長崎",
    "ロアッソ熊本": "熊本",
    "大分トリニータ": "大分",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_team(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", "", text).replace("･", "・")
    return TEAM_ALIASES.get(text, text)


def normalize_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").fillna("")


def find_column(frame: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        matched = lower_map.get(candidate.lower())
        if matched:
            return matched
    if required:
        raise ValueError(f"Missing one of columns: {candidates}")
    return None


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{label} is empty: {path}")
    return frame


def prepare_training(frame: pd.DataFrame) -> pd.DataFrame:
    home_col = find_column(frame, ["homeTeam", "home_team", "home"])
    away_col = find_column(frame, ["awayTeam", "away_team", "away"])
    id_col = find_column(frame, ["match_card_id", "matchId", "match_id"], required=False)
    date_col = find_column(frame, ["matchDate", "match_date", "date"], required=False)
    competition_col = find_column(frame, ["competition", "league", "category"], required=False)

    result = frame.copy()
    result["home_raw"] = result[home_col].astype(str)
    result["away_raw"] = result[away_col].astype(str)
    result["home_key"] = result[home_col].map(normalize_team)
    result["away_key"] = result[away_col].map(normalize_team)
    result["date_key"] = normalize_date(result[date_col]) if date_col else ""
    result["competition_key"] = result[competition_col].astype(str) if competition_col else ""
    result["match_id_key"] = (
        pd.to_numeric(result[id_col], errors="coerce").astype("Int64")
        if id_col
        else pd.Series(pd.NA, index=result.index, dtype="Int64")
    )
    result["row_id"] = range(len(result))
    return result


def prepare_matches(frame: pd.DataFrame) -> pd.DataFrame:
    id_col = find_column(frame, ["match_card_id", "matchId", "match_id"])
    home_col = find_column(frame, ["home_team", "homeTeam", "home"])
    away_col = find_column(frame, ["away_team", "awayTeam", "away"])
    date_col = find_column(frame, ["match_date", "matchDate", "date"], required=False)
    competition_col = find_column(frame, ["competition", "league", "category"], required=False)

    result = pd.DataFrame({
        "match_id_key": pd.to_numeric(frame[id_col], errors="coerce").astype("Int64"),
        "home_key": frame[home_col].map(normalize_team),
        "away_key": frame[away_col].map(normalize_team),
        "date_key": normalize_date(frame[date_col]) if date_col else "",
        "competition_key": frame[competition_col].astype(str) if competition_col else "",
    })
    return result.dropna(subset=["match_id_key"]).drop_duplicates("match_id_key", keep="last")


def classify_rows(training: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["home_key", "away_key"]
    use_date = training["date_key"].ne("").any() and matches["date_key"].ne("").any()
    if use_date:
        key_columns.append("date_key")

    key_lookup = matches[key_columns + ["match_id_key"]].drop_duplicates(key_columns, keep=False)
    key_merged = training[key_columns].merge(key_lookup, on=key_columns, how="left")

    available_teams = set(matches["home_key"]) | set(matches["away_key"])
    reversed_pairs = set(zip(matches["away_key"], matches["home_key"], strict=False))

    reasons: list[str] = []
    resolved_ids: list[object] = []

    for i, row in training.iterrows():
        match_id = row["match_id_key"]
        if pd.notna(match_id) and (matches["match_id_key"] == match_id).any():
            reasons.append("matched_by_match_id")
            resolved_ids.append(int(match_id))
            continue

        candidate_id = key_merged.iloc[i]["match_id_key"]
        if pd.notna(candidate_id):
            reasons.append("matched_by_team_date")
            resolved_ids.append(int(candidate_id))
            continue

        home_key = row["home_key"]
        away_key = row["away_key"]

        if home_key not in available_teams or away_key not in available_teams:
            reason = "team_not_in_player_data"
        elif (home_key, away_key) in reversed_pairs:
            reason = "home_away_reversed"
        elif row["date_key"] == "":
            reason = "missing_training_date"
        else:
            same_pair = matches.loc[(matches["home_key"] == home_key) & (matches["away_key"] == away_key)]
            if not same_pair.empty:
                reason = "date_mismatch"
            else:
                competition = str(row["competition_key"]).lower()
                if any(k in competition for k in ("j2", "j3", "海外", "premier", "cup")):
                    reason = "competition_not_collected"
                else:
                    reason = "unknown"

        reasons.append(reason)
        resolved_ids.append(pd.NA)

    result = training[[
        "row_id", "home_raw", "away_raw", "home_key", "away_key",
        "date_key", "competition_key", "match_id_key",
    ]].copy()
    result["diagnosis"] = reasons
    result["resolved_match_id"] = pd.Series(resolved_ids, dtype="Int64")
    return result


def build_team_report(diagnosis: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    available = set(matches["home_key"]) | set(matches["away_key"])
    counter: Counter[tuple[str, str]] = Counter()
    unmatched = diagnosis.loc[~diagnosis["diagnosis"].str.startswith("matched_")]

    for row in unmatched.itertuples(index=False):
        if row.home_key not in available:
            counter[(row.home_raw, row.home_key)] += 1
        if row.away_key not in available:
            counter[(row.away_raw, row.away_key)] += 1

    return pd.DataFrame([
        {"training_team": raw, "normalized_team": normalized, "unmatched_rows": count}
        for (raw, normalized), count in counter.most_common()
    ])


def parse_args() -> argparse.Namespace:
    root = project_root()
    processed = root / "ml" / "player_engine" / "data" / "processed"
    parser = argparse.ArgumentParser(description="Diagnose Player Intelligence matching failures.")
    parser.add_argument("--training", type=Path, default=root / "toto-training-dataset.csv")
    parser.add_argument("--matches", type=Path, default=processed / "matches.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "ml" / "diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = prepare_training(load_csv(args.training, "Training dataset"))
    matches = prepare_matches(load_csv(args.matches, "Parsed matches"))

    diagnosis = classify_rows(training, matches)
    team_report = build_team_report(diagnosis, matches)

    summary = diagnosis["diagnosis"].value_counts(dropna=False).rename_axis("reason").reset_index(name="rows")
    summary["rate"] = summary["rows"] / len(diagnosis)
    competition_report = (
        diagnosis.groupby(["competition_key", "diagnosis"], dropna=False)
        .size().reset_index(name="rows").sort_values("rows", ascending=False)
    )

    matched = int(diagnosis["diagnosis"].str.startswith("matched_").sum())
    total = len(diagnosis)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnosis.to_csv(args.output_dir / "player_matching_diagnosis.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "player_matching_summary.csv", index=False, encoding="utf-8-sig")
    team_report.to_csv(args.output_dir / "unmatched_team_names.csv", index=False, encoding="utf-8-sig")
    competition_report.to_csv(args.output_dir / "competition_matching_report.csv", index=False, encoding="utf-8-sig")

    report = {
        "training_rows": total,
        "matched_rows": matched,
        "unmatched_rows": total - matched,
        "match_rate": matched / total if total else 0.0,
        "reasons": {str(row.reason): int(row.rows) for row in summary.itertuples(index=False)},
    }
    (args.output_dir / "player_matching_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("Player Matching Diagnosis")
    print("=" * 72)
    print(f"Training rows : {total}")
    print(f"Matched       : {matched}")
    print(f"Unmatched     : {total - matched}")
    print(f"Match rate    : {matched / total:.2%}")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Saved to      : {args.output_dir}")


if __name__ == "__main__":
    main()
