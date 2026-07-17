from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

J1_TEAMS = {"札幌","鹿島","浦和","柏","FC東京","東京V","川崎F","横浜FM","横浜FC","湘南","新潟","清水","名古屋","京都","G大阪","C大阪","神戸","岡山","広島","福岡","鳥栖","町田","磐田"}
J2_TEAMS = {"仙台","秋田","山形","いわき","水戸","栃木","群馬","千葉","甲府","富山","藤枝","山口","徳島","愛媛","長崎","熊本","大分","大宮"}
J3_TEAMS = {"八戸","岩手","福島","YS横浜","相模原","松本","長野","金沢","沼津","岐阜","FC大阪","奈良","鳥取","讃岐","今治","北九州","宮崎","琉球"}

TEAM_ALIASES = {
    "北海道コンサドーレ札幌":"札幌","コンサドーレ札幌":"札幌","鹿島アントラーズ":"鹿島","浦和レッズ":"浦和","柏レイソル":"柏",
    "ＦＣ東京":"FC東京","FC東京":"FC東京","東京ヴェルディ":"東京V","東京Ｖ":"東京V","川崎フロンターレ":"川崎F","川崎Ｆ":"川崎F",
    "横浜F・マリノス":"横浜FM","横浜Ｆ・マリノス":"横浜FM","横浜ＦＣ":"横浜FC","横浜FC":"横浜FC","湘南ベルマーレ":"湘南",
    "アルビレックス新潟":"新潟","清水エスパルス":"清水","名古屋グランパス":"名古屋","京都サンガF.C.":"京都","京都サンガＦ．Ｃ．":"京都",
    "ガンバ大阪":"G大阪","Ｇ大阪":"G大阪","セレッソ大阪":"C大阪","Ｃ大阪":"C大阪","ヴィッセル神戸":"神戸","ファジアーノ岡山":"岡山",
    "サンフレッチェ広島":"広島","アビスパ福岡":"福岡","サガン鳥栖":"鳥栖","ＦＣ町田ゼルビア":"町田","FC町田ゼルビア":"町田","町田ゼルビア":"町田",
    "ベガルタ仙台":"仙台","ブラウブリッツ秋田":"秋田","モンテディオ山形":"山形","いわきＦＣ":"いわき","いわきFC":"いわき","水戸ホーリーホック":"水戸",
    "栃木ＳＣ":"栃木","栃木SC":"栃木","ザスパ群馬":"群馬","ジェフユナイテッド千葉":"千葉","ヴァンフォーレ甲府":"甲府","カターレ富山":"富山",
    "ジュビロ磐田":"磐田","藤枝ＭＹＦＣ":"藤枝","藤枝MYFC":"藤枝","レノファ山口ＦＣ":"山口","レノファ山口FC":"山口","徳島ヴォルティス":"徳島",
    "愛媛ＦＣ":"愛媛","愛媛FC":"愛媛","Ｖ・ファーレン長崎":"長崎","V・ファーレン長崎":"長崎","ロアッソ熊本":"熊本","大分トリニータ":"大分",
    "RB大宮アルディージャ":"大宮","大宮アルディージャ":"大宮"
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_team(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", "", text).replace("･", "・")
    return TEAM_ALIASES.get(text, text)


def find_column(frame: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        match = lower_map.get(candidate.lower())
        if match:
            return match
    if required:
        raise ValueError(f"Missing one of columns: {candidates}")
    return None


def classify_team(team: str) -> str:
    if team in J1_TEAMS:
        return "J1"
    if team in J2_TEAMS:
        return "J2"
    if team in J3_TEAMS:
        return "J3"
    return "Unknown" if not team else "Other"


def classify_match(home: str, away: str) -> str:
    home_class, away_class = classify_team(home), classify_team(away)
    if home_class == away_class:
        return home_class
    pair = {home_class, away_class}
    if pair <= {"J1", "J2", "J3"}:
        return "Mixed Domestic"
    if "Other" in pair:
        return "Other"
    return "Unknown"


def analyze_training_dataset(frame: pd.DataFrame):
    home_col = find_column(frame, ["homeTeam", "home_team", "home"])
    away_col = find_column(frame, ["awayTeam", "away_team", "away"])
    round_col = find_column(frame, ["round", "roundNo", "totoRound", "toto_round"], required=False)
    date_col = find_column(frame, ["date", "matchDate", "match_date"], required=False)
    competition_col = find_column(frame, ["competition", "league", "category"], required=False)

    result = frame.copy()
    result["home_raw"] = result[home_col].astype(str)
    result["away_raw"] = result[away_col].astype(str)
    result["home_team"] = result[home_col].map(normalize_team)
    result["away_team"] = result[away_col].map(normalize_team)
    result["home_class"] = result["home_team"].map(classify_team)
    result["away_class"] = result["away_team"].map(classify_team)
    result["match_class"] = [classify_match(h, a) for h, a in zip(result["home_team"], result["away_team"], strict=False)]
    result["toto_round"] = pd.to_numeric(result[round_col], errors="coerce").astype("Int64") if round_col else pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["match_date"] = pd.to_datetime(result[date_col], errors="coerce").dt.strftime("%Y-%m-%d") if date_col else ""
    result["competition_raw"] = result[competition_col].astype(str) if competition_col else ""

    match_summary = result["match_class"].value_counts(dropna=False).rename_axis("match_class").reset_index(name="rows")
    match_summary["rate"] = match_summary["rows"] / len(result)

    team_counter: Counter[tuple[str, str]] = Counter()
    for row in result.itertuples(index=False):
        team_counter[(row.home_raw, row.home_team)] += 1
        team_counter[(row.away_raw, row.away_team)] += 1

    team_report = pd.DataFrame([
        {"raw_team_name": raw, "normalized_team": norm, "classified_as": classify_team(norm), "appearances": count}
        for (raw, norm), count in team_counter.most_common()
    ])
    unknown_teams = team_report.loc[team_report["classified_as"].isin(["Other", "Unknown"])].copy()
    round_report = result.groupby(["toto_round", "match_class"], dropna=False).size().reset_index(name="rows").sort_values(["toto_round", "rows"], ascending=[True, False])

    summary = {
        "training_rows": len(result),
        "unique_toto_rounds": int(result["toto_round"].nunique(dropna=True)),
        "round_from": int(result["toto_round"].min()) if result["toto_round"].notna().any() else None,
        "round_to": int(result["toto_round"].max()) if result["toto_round"].notna().any() else None,
        "unique_normalized_teams": int(pd.concat([result["home_team"], result["away_team"]], ignore_index=True).nunique()),
        "match_class_counts": {str(row.match_class): int(row.rows) for row in match_summary.itertuples(index=False)},
    }

    detail_columns = ["toto_round","match_date","home_raw","away_raw","home_team","away_team","home_class","away_class","match_class","competition_raw"]
    return result[detail_columns], match_summary, team_report, unknown_teams, round_report, summary


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Analyze the composition of toto training data.")
    parser.add_argument("--input", type=Path, default=root / "toto-training-dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "ml" / "diagnostics" / "training_dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Training dataset not found: {args.input}")
    frame = pd.read_csv(args.input)
    if frame.empty:
        raise ValueError(f"Training dataset is empty: {args.input}")

    detail, match_summary, team_report, unknown_teams, round_report, summary = analyze_training_dataset(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "training_match_classification.csv", index=False, encoding="utf-8-sig")
    match_summary.to_csv(args.output_dir / "training_match_summary.csv", index=False, encoding="utf-8-sig")
    team_report.to_csv(args.output_dir / "training_team_report.csv", index=False, encoding="utf-8-sig")
    unknown_teams.to_csv(args.output_dir / "training_unknown_teams.csv", index=False, encoding="utf-8-sig")
    round_report.to_csv(args.output_dir / "training_round_report.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "training_dataset_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Training Dataset Analysis")
    print("=" * 72)
    print(f"Training rows        : {summary['training_rows']}")
    print(f"Unique toto rounds   : {summary['unique_toto_rounds']}")
    print(f"Round range          : {summary['round_from']} - {summary['round_to']}")
    print(f"Unique teams         : {summary['unique_normalized_teams']}")
    print()
    print(match_summary.to_string(index=False))
    print()
    print(f"Unknown team names   : {len(unknown_teams)}")
    print(f"Saved to             : {args.output_dir}")


if __name__ == "__main__":
    main()
