from __future__ import annotations

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_env_local(path: Path) -> None:
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


def normalize_team(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.replace("・", "").replace("･", "")
    text = re.sub(r"[\s\-_／/()（）\[\]［］.]+", "", text)

    aliases = {
        "横浜fマリノス": "横浜fm",
        "横浜fm": "横浜fm",
        "川崎フロンターレ": "川崎f",
        "川崎f": "川崎f",
        "浦和レッズ": "浦和",
        "鹿島アントラーズ": "鹿島",
        "柏レイソル": "柏",
        "アルビレックス新潟": "新潟",
        "サンフレッチェ広島": "広島",
        "ガンバ大阪": "g大阪",
        "セレッソ大阪": "c大阪",
        "東京ヴェルディ": "東京v",
        "名古屋グランパス": "名古屋",
        "京都サンガfc": "京都",
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


def supabase_select(table: str, query_params: dict[str, str]) -> list[dict[str, Any]]:
    url = (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SECRET_KEY")
        or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        or ""
    ).strip()

    if not url or not key:
        raise RuntimeError("Supabase environment variables are missing.")

    response = requests.get(
        f"{url.rstrip('/')}/rest/v1/{table}",
        params=query_params,
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Supabase fetch failed: {response.status_code} {response.text}"
        )
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Supabase response: {payload}")
    return payload


def closest_names(source: str, candidates: list[str], limit: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            {
                "candidate": candidate,
                "score": round(
                    SequenceMatcher(
                        None,
                        normalize_team(source),
                        normalize_team(candidate),
                    ).ratio(),
                    4,
                ),
            }
            for candidate in candidates
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return ranked[:limit]


def main() -> None:
    root = project_root()
    load_env_local(root / ".env.local")

    rf_path = root / "ml" / "rf_v6" / "randomforest_v6_predictions.csv"
    rf = pd.read_csv(rf_path)
    rf["match_date"] = pd.to_datetime(rf["match_date"], errors="coerce")
    rf["home_key"] = rf["home_team"].map(normalize_team)
    rf["away_key"] = rf["away_team"].map(normalize_team)
    rf["pair_key"] = rf["home_key"] + "||" + rf["away_key"]

    fixture_rows = supabase_select(
        "toto_round_fixtures",
        {
            "select": "round_no,match_no,home_team,away_team,kickoff_at,toto_result",
            "round_no": "gte.1618",
            "order": "round_no.asc,match_no.asc",
        },
    )
    fixtures = pd.DataFrame(fixture_rows)
    if fixtures.empty:
        raise RuntimeError("No fixture rows found in Supabase.")

    fixtures["kickoff_at"] = pd.to_datetime(
        fixtures["kickoff_at"], errors="coerce", format="mixed"
    )
    fixtures["home_key"] = fixtures["home_team"].map(normalize_team)
    fixtures["away_key"] = fixtures["away_team"].map(normalize_team)
    fixtures["pair_key"] = fixtures["home_key"] + "||" + fixtures["away_key"]

    prediction_pairs = set(rf["pair_key"].dropna())
    fixture_pairs = set(fixtures["pair_key"].dropna())
    exact_overlap = sorted(prediction_pairs & fixture_pairs)

    prediction_teams = sorted(
        set(rf["home_team"].dropna().astype(str))
        | set(rf["away_team"].dropna().astype(str))
    )
    fixture_teams = sorted(
        set(fixtures["home_team"].dropna().astype(str))
        | set(fixtures["away_team"].dropna().astype(str))
    )
    prediction_keys = {normalize_team(item) for item in prediction_teams}
    unmatched_fixture_teams = sorted(
        team for team in fixture_teams if normalize_team(team) not in prediction_keys
    )
    fuzzy = {
        team: closest_names(team, prediction_teams)
        for team in unmatched_fixture_teams[:50]
    }

    report = {
        "rf_rows": int(len(rf)),
        "rf_date_min": rf["match_date"].min().isoformat()
        if rf["match_date"].notna().any()
        else None,
        "rf_date_max": rf["match_date"].max().isoformat()
        if rf["match_date"].notna().any()
        else None,
        "fixture_rows": int(len(fixtures)),
        "fixture_round_min": int(fixtures["round_no"].min()),
        "fixture_round_max": int(fixtures["round_no"].max()),
        "fixture_date_min": fixtures["kickoff_at"].min().isoformat()
        if fixtures["kickoff_at"].notna().any()
        else None,
        "fixture_date_max": fixtures["kickoff_at"].max().isoformat()
        if fixtures["kickoff_at"].notna().any()
        else None,
        "prediction_unique_pairs": len(prediction_pairs),
        "fixture_unique_pairs": len(fixture_pairs),
        "exact_pair_overlap_count": len(exact_overlap),
        "exact_pair_overlap_sample": exact_overlap[:20],
        "unmatched_fixture_team_count": len(unmatched_fixture_teams),
        "unmatched_fixture_teams": unmatched_fixture_teams,
        "fuzzy_candidates": fuzzy,
    }

    output_dir = root / "ml" / "true_backtest_v6" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "v6_true_backtest_matching_diagnosis.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "fixture_team": team,
                "fixture_key": normalize_team(team),
                "best_prediction_team": values[0]["candidate"] if values else "",
                "best_score": values[0]["score"] if values else None,
            }
            for team, values in fuzzy.items()
        ]
    ).to_csv(
        output_dir / "v6_true_backtest_team_matching.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 92)
    print("Project Alpha v6 True Backtest Matching Diagnosis")
    print("=" * 92)
    print(f"RF rows                     : {report['rf_rows']}")
    print(f"RF date range               : {report['rf_date_min']} -> {report['rf_date_max']}")
    print(f"Fixture rows                : {report['fixture_rows']}")
    print(f"Fixture round range         : {report['fixture_round_min']} -> {report['fixture_round_max']}")
    print(f"Fixture date range          : {report['fixture_date_min']} -> {report['fixture_date_max']}")
    print(f"Exact pair overlap          : {report['exact_pair_overlap_count']}")
    print(f"Unmatched fixture teams     : {report['unmatched_fixture_team_count']}")
    print(f"Diagnostics                 : {json_path}")


if __name__ == "__main__":
    main()
