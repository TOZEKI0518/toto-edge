from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger("historical_toto_fixture_backfill")

RESULT_URL = (
    "https://store.toto-dream.com/dcs/subos/screen/pi04/spin011/"
    "PGSPIN01101LnkHoldCntLotResultLsttoto.form"
)

FULL_DATE_PATTERNS = (
    re.compile(r"(20\d{2})[年/\-.]\s*(\d{1,2})[月/\-.]\s*(\d{1,2})日?"),
    re.compile(r"(20\d{2})(\d{2})(\d{2})"),
)


@dataclass(frozen=True, slots=True)
class Config:
    round_from: int
    round_to: int
    target_year_from: int
    target_year_to: int
    output_dir: Path
    timeout_seconds: float = 30.0
    sleep_seconds: float = 0.35
    retries: int = 3
    save_html: bool = False
    dry_run: bool = False
    force: bool = False

    def validate(self) -> None:
        if self.round_from <= 0 or self.round_to <= 0:
            raise ValueError("round numbers must be positive.")
        if self.target_year_from > self.target_year_to:
            raise ValueError("target year range is invalid.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.sleep_seconds < 0:
            raise ValueError("sleep_seconds must not be negative.")
        if self.retries < 1:
            raise ValueError("retries must be at least 1.")


class BackfillError(RuntimeError):
    """Raised when historical toto backfill cannot continue safely."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_env_local(path: Path) -> None:
    """Load KEY=VALUE pairs without overriding active shell variables."""
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


def normalize_space(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", text).strip()


def normalize_team(value: str) -> str:
    return re.sub(r"\s+", "", normalize_space(value))


def parse_full_date(text: str) -> date | None:
    normalized = unicodedata.normalize("NFKC", text)

    for pattern in FULL_DATE_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue

        year, month, day = map(int, match.groups())

        try:
            return date(year, month, day)
        except ValueError:
            continue

    return None


def parse_month_day(text: str) -> tuple[int, int] | None:
    normalized = unicodedata.normalize("NFKC", text)
    match = re.search(r"(?<!\d)(\d{1,2})[月/.-](\d{1,2})日?", normalized)
    if not match:
        return None

    month, day = map(int, match.groups())
    try:
        date(2000, month, day)
    except ValueError:
        return None
    return month, day


def outcome_from_cell(value: str) -> str | None:
    normalized = normalize_space(value)
    return normalized if normalized in {"0", "1", "2"} else None


def extract_round_date(html: str) -> date | None:
    soup = BeautifulSoup(html, "html.parser")

    candidates = [
        soup.get_text(" ", strip=True),
        *(element.get_text(" ", strip=True) for element in soup.select("h1,h2,h3,caption,th")),
    ]

    for candidate in candidates:
        parsed = parse_full_date(candidate)
        if parsed is not None:
            return parsed

    return None


def find_product_label(table: Any) -> str:
    container = table.find_parent("div", class_="kujikekka")
    if container is None:
        return "unknown"

    heading_text = normalize_space(container.get_text(" ", strip=True))
    if "mini toto" in heading_text:
        return "mini_toto"
    if "totoGOAL" in heading_text:
        return "toto_goal"
    if re.search(r"\btoto\b", heading_text, flags=re.IGNORECASE):
        return "toto"
    return "unknown"


def classify_competition(home_team: str, away_team: str) -> str:
    jleague_keys = {
        "札幌", "八戸", "岩手", "仙台", "秋田", "山形", "福島", "いわき",
        "水戸", "栃木", "栃木C", "群馬", "鹿島", "浦和", "大宮", "千葉",
        "柏", "FC東京", "F東京", "東京V", "町田", "川崎F", "川崎",
        "横浜FM", "横浜M", "横浜FC", "YS横浜", "湘南", "相模原", "甲府",
        "松本", "長野", "新潟", "富山", "金沢", "清水", "磐田", "藤枝",
        "沼津", "名古屋", "岐阜", "京都", "G大阪", "C大阪", "FC大阪",
        "神戸", "奈良", "滋賀", "鳥取", "岡山", "広島", "山口", "徳島",
        "愛媛", "今治", "讃岐", "高知", "福岡", "北九州", "鳥栖",
        "長崎", "熊本", "大分", "宮崎", "鹿児島", "琉球",
    }
    return (
        "J.League"
        if home_team in jleague_keys and away_team in jleague_keys
        else "Other"
    )


def parse_fixtures(
    html: str,
    round_no: int,
) -> tuple[date | None, list[dict[str, Any]]]:
    """
    Parse toto / mini toto result tables only.

    Expected columns:
        開催日, 競技場, No, ホーム, 試合結果, アウェイ, くじ結果
    """
    soup = BeautifulSoup(html, "html.parser")
    round_date = extract_round_date(html)
    fixtures: dict[tuple[str, int], dict[str, Any]] = {}

    for table in soup.select("table.kobetsu-format2"):
        header_text = normalize_space(
            " ".join(
                cell.get_text(" ", strip=True)
                for cell in table.select("th")
            )
        )

        required_headers = ("ホーム", "試合結果", "アウェイ", "くじ結果")
        if not all(header in header_text for header in required_headers):
            continue

        product_type = find_product_label(table)
        if product_type == "toto_goal":
            continue

        for row in table.select("tr"):
            cells = [
                normalize_space(cell.get_text(" ", strip=True))
                for cell in row.select("td")
            ]
            if len(cells) < 7:
                continue

            if not re.fullmatch(r"\d{1,2}", cells[2]):
                continue

            match_no = int(cells[2])
            if not 1 <= match_no <= 13:
                continue

            match_day_text = cells[0]
            venue = cells[1] or None
            home_team = normalize_team(cells[3])
            score_text = cells[4]
            away_team = normalize_team(cells[5])
            toto_result = outcome_from_cell(cells[6])

            if not home_team or not away_team or toto_result is None:
                continue
            if not re.fullmatch(r"\d+\s*-\s*\d+", score_text):
                continue

            row_month_day = parse_month_day(match_day_text)
            kickoff_date: date | None = round_date

            if round_date is not None and row_month_day is not None:
                month, day = row_month_day
                candidates = [
                    date(year, month, day)
                    for year in (
                        round_date.year - 1,
                        round_date.year,
                        round_date.year + 1,
                    )
                    if _is_valid_date(year, month, day)
                ]
                if candidates:
                    kickoff_date = min(
                        candidates,
                        key=lambda value: abs((value - round_date).days),
                    )

            fixtures[(product_type, match_no)] = {
                "round_no": round_no,
                "match_no": match_no,
                "home_team": home_team,
                "away_team": away_team,
                "kickoff_at": (
                    kickoff_date.isoformat()
                    if kickoff_date is not None
                    else None
                ),
                "venue": venue,
                "toto_result": toto_result,
                "_product_type": product_type,
                "_competition": classify_competition(
                    home_team,
                    away_team,
                ),
            }

    return round_date, [
        fixtures[key]
        for key in sorted(
            fixtures,
            key=lambda value: (value[0], value[1]),
        )
    ]


def _is_valid_date(year: int, month: int, day: int) -> bool:
    try:
        date(year, month, day)
        return True
    except ValueError:
        return False


class TotoClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/150 Safari/537.36"
                )
            }
        )

    def fetch_round(self, round_no: int) -> tuple[str, str]:
        params = {
            "popupDispDiv": "disp",
            "holdCntId": str(round_no),
        }

        last_error: Exception | None = None
        for attempt in range(1, self.config.retries + 1):
            try:
                response = self.session.get(
                    RESULT_URL,
                    params=params,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                response.encoding = response.apparent_encoding or response.encoding
                return response.text, response.url
            except requests.RequestException as error:
                last_error = error
                LOGGER.warning(
                    "Round %d fetch failed (%d/%d): %s",
                    round_no,
                    attempt,
                    self.config.retries,
                    error,
                )
                if attempt < self.config.retries:
                    time.sleep(min(2**attempt, 8))

        raise BackfillError(
            f"Round {round_no} fetch failed after retries: {last_error}"
        )


class SupabaseRepository:
    def __init__(self, timeout_seconds: float) -> None:
        url = (
            os.getenv("SUPABASE_URL")
            or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
            or ""
        ).strip()
        key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_SECRET_KEY")
            or ""
        ).strip()

        if not url:
            raise BackfillError("SUPABASE_URL is not configured.")
        if not key:
            raise BackfillError("SUPABASE_SERVICE_ROLE_KEY is not configured.")

        self.base_url = url.rstrip("/") + "/rest/v1"
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
        )

    def complete_rounds(self, round_numbers: list[int]) -> set[int]:
        if not round_numbers:
            return set()

        completed: set[int] = set()

        for start in range(0, len(round_numbers), 80):
            chunk = round_numbers[start : start + 80]
            response = self.session.get(
                f"{self.base_url}/toto_round_fixtures",
                params={
                    "select": "round_no,match_no,kickoff_at",
                    "round_no": f"in.({','.join(map(str, chunk))})",
                    "order": "round_no.asc,match_no.asc",
                },
                timeout=self.timeout_seconds,
            )
            if response.status_code != 200:
                raise BackfillError(
                    "Supabase existing-row check failed: "
                    f"{response.status_code} {response.text}"
                )

            frame = pd.DataFrame(response.json())
            if frame.empty:
                continue

            for round_no, group in frame.groupby("round_no"):
                valid_dates = pd.to_datetime(
                    group["kickoff_at"],
                    errors="coerce",
                    format="mixed",
                )
                if len(group) >= 11 and (valid_dates.dt.year >= 2000).all():
                    completed.add(int(round_no))

        return completed

    def upsert_fixtures(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        database_rows = [
            {
                key: value
                for key, value in row.items()
                if not key.startswith("_")
            }
            for row in rows
        ]

        response = self.session.post(
            f"{self.base_url}/toto_round_fixtures",
            params={"on_conflict": "round_no,match_no"},
            headers={
                **self.session.headers,
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            json=database_rows,
            timeout=self.timeout_seconds,
        )
        if response.status_code not in {200, 201, 204}:
            raise BackfillError(
                "Supabase fixture upsert failed: "
                f"{response.status_code} {response.text}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill historical toto fixtures with full dates for "
            "Project Alpha v6 true backtesting."
        )
    )
    parser.add_argument("--round-from", type=int, default=1200)
    parser.add_argument("--round-to", type=int, default=1617)
    parser.add_argument("--year-from", type=int, default=2022)
    parser.add_argument("--year-to", type=int, default=2025)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "ml" / "historical_toto_backfill",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--save-html", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_env_local(project_root() / ".env.local")
    args = parse_args()
    configure_logging(args.verbose)

    config = Config(
        round_from=min(args.round_from, args.round_to),
        round_to=max(args.round_from, args.round_to),
        target_year_from=min(args.year_from, args.year_to),
        target_year_to=max(args.year_from, args.year_to),
        output_dir=args.output_dir,
        timeout_seconds=args.timeout,
        sleep_seconds=args.sleep,
        retries=args.retries,
        save_html=args.save_html,
        dry_run=args.dry_run,
        force=args.force,
    )
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    round_numbers = list(
        range(config.round_from, config.round_to + 1)
    )
    repository = (
        None
        if config.dry_run
        else SupabaseRepository(config.timeout_seconds)
    )
    completed = (
        set()
        if config.force or repository is None
        else repository.complete_rounds(round_numbers)
    )

    client = TotoClient(config)
    all_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for round_no in round_numbers:
        if round_no in completed:
            diagnostics.append(
                {
                    "round_no": round_no,
                    "status": "skipped_complete",
                }
            )
            continue

        try:
            html, source_url = client.fetch_round(round_no)
            round_date, fixtures = parse_fixtures(html, round_no)

            if config.save_html:
                (config.output_dir / "html").mkdir(
                    parents=True,
                    exist_ok=True,
                )
                (
                    config.output_dir
                    / "html"
                    / f"toto_round_{round_no}.html"
                ).write_text(html, encoding="utf-8")

            if round_date is None:
                diagnostics.append(
                    {
                        "round_no": round_no,
                        "status": "no_full_date",
                        "fixture_count": len(fixtures),
                        "source_url": source_url,
                    }
                )
                continue

            if not (
                config.target_year_from
                <= round_date.year
                <= config.target_year_to
            ):
                diagnostics.append(
                    {
                        "round_no": round_no,
                        "status": "outside_target_year",
                        "round_date": round_date.isoformat(),
                        "fixture_count": len(fixtures),
                    }
                )
                continue

            if len(fixtures) < 5:
                diagnostics.append(
                    {
                        "round_no": round_no,
                        "status": "insufficient_fixtures",
                        "round_date": round_date.isoformat(),
                        "fixture_count": len(fixtures),
                    }
                )
                continue

            if repository is not None:
                repository.upsert_fixtures(fixtures)

            all_rows.extend(fixtures)
            diagnostics.append(
                {
                    "round_no": round_no,
                    "status": (
                        "dry_run_ready"
                        if config.dry_run
                        else "uploaded"
                    ),
                    "round_date": round_date.isoformat(),
                    "fixture_count": len(fixtures),
                    "source_url": source_url,
                }
            )
            LOGGER.info(
                "Round %d | date=%s fixtures=%d status=%s",
                round_no,
                round_date,
                len(fixtures),
                diagnostics[-1]["status"],
            )

        except Exception as error:  # keep long backfill resumable
            LOGGER.exception("Round %d failed.", round_no)
            diagnostics.append(
                {
                    "round_no": round_no,
                    "status": "error",
                    "error": str(error),
                }
            )

        if config.sleep_seconds:
            time.sleep(config.sleep_seconds)

    rows_frame = pd.DataFrame(all_rows)
    if not rows_frame.empty:
        rows_frame = rows_frame.rename(
            columns={
                "_product_type": "product_type",
                "_competition": "competition",
            }
        )
    diagnostics_frame = pd.DataFrame(diagnostics)

    rows_frame.to_csv(
        config.output_dir / "historical_toto_fixtures_backfilled.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics_frame.to_csv(
        config.output_dir / "historical_toto_backfill_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "round_from": config.round_from,
        "round_to": config.round_to,
        "target_year_from": config.target_year_from,
        "target_year_to": config.target_year_to,
        "rounds_scanned": len(round_numbers),
        "rounds_uploaded": int(
            (diagnostics_frame.get("status") == "uploaded").sum()
        )
        if not diagnostics_frame.empty
        else 0,
        "rounds_dry_run_ready": int(
            (diagnostics_frame.get("status") == "dry_run_ready").sum()
        )
        if not diagnostics_frame.empty
        else 0,
        "fixture_rows": len(rows_frame),
        "output_dir": str(config.output_dir),
        "completed_at": datetime.now().isoformat(),
    }
    (
        config.output_dir / "historical_toto_backfill_summary.json"
    ).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 92)
    print("Project Alpha Historical Toto Fixture Backfill")
    print("=" * 92)
    print(f"Rounds scanned             : {summary['rounds_scanned']}")
    print(f"Rounds uploaded            : {summary['rounds_uploaded']}")
    print(f"Dry-run ready rounds       : {summary['rounds_dry_run_ready']}")
    print(f"Fixture rows               : {summary['fixture_rows']}")
    print(f"Output                     : {config.output_dir}")


if __name__ == "__main__":
    main()
