from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup, Tag


LOGGER = logging.getLogger("jleague_match_id_collector")

BASE_URL = "https://data.j-league.or.jp"
SEARCH_URL = f"{BASE_URL}/SFMS01/search"
MATCH_ID_PATTERN = re.compile(r"match_card_id=(\d+)")


@dataclass(frozen=True)
class MatchIndexRow:
    match_card_id: str
    detail_url: str
    season: str = ""
    competition: str = ""
    section: str = ""
    match_date: str = ""
    kickoff: str = ""
    home_team: str = ""
    score: str = ""
    away_team: str = ""
    stadium: str = ""
    attendance: str = ""


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "match_card_ids.csv"
    )


def build_search_url(
    years: list[str],
    competition_frame_ids: list[str],
) -> str:
    params: list[tuple[str, str]] = []

    for year in years:
        params.append(("competition_years", year))

    for frame_id in competition_frame_ids:
        params.append(("competition_frame_ids", frame_id))

    if not params:
        raise ValueError(
            "Specify at least one --year or --competition-frame-id, "
            "or use --search-url."
        )

    return f"{SEARCH_URL}?{urlencode(params)}"


def fetch_html(url: str, timeout: int = 30) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/150.0 Safari/537.36 "
                "Toto-Quant-Project-Alpha/1.0"
            ),
            "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
        },
    )

    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
        encoding = response.headers.get_content_charset() or "utf-8"

    return raw.decode(encoding, errors="replace")


def clean_text(tag: Tag | None) -> str:
    if tag is None:
        return ""
    return " ".join(tag.get_text(" ", strip=True).split())


def row_from_table(anchor: Tag, match_card_id: str) -> MatchIndexRow:
    tr = anchor.find_parent("tr")
    detail_url = urljoin(BASE_URL, str(anchor.get("href", "")))

    if tr is None:
        return MatchIndexRow(
            match_card_id=match_card_id,
            detail_url=detail_url,
        )

    cells = [clean_text(td) for td in tr.find_all("td")]

    # Official result table normally contains:
    # season, competition, section, date, kickoff, home, score,
    # away, stadium, attendance, broadcast.
    padded = cells + [""] * max(0, 11 - len(cells))

    return MatchIndexRow(
        match_card_id=match_card_id,
        detail_url=detail_url,
        season=padded[0],
        competition=padded[1],
        section=padded[2],
        match_date=padded[3],
        kickoff=padded[4],
        home_team=padded[5],
        score=padded[6],
        away_team=padded[7],
        stadium=padded[8],
        attendance=padded[9],
    )


def parse_match_rows(html: str) -> list[MatchIndexRow]:
    soup = BeautifulSoup(html, "lxml")
    rows: dict[str, MatchIndexRow] = {}

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        match = MATCH_ID_PATTERN.search(href)
        if match is None:
            continue

        match_card_id = match.group(1)
        rows[match_card_id] = row_from_table(anchor, match_card_id)

    # Fallback for IDs embedded in JavaScript or uncommon markup.
    for match_card_id in MATCH_ID_PATTERN.findall(html):
        rows.setdefault(
            match_card_id,
            MatchIndexRow(
                match_card_id=match_card_id,
                detail_url=(
                    f"{BASE_URL}/SFMS02/?match_card_id={match_card_id}"
                ),
            ),
        )

    return sorted(rows.values(), key=lambda row: int(row.match_card_id))


def merge_existing(
    new_rows: Iterable[MatchIndexRow],
    output_path: Path,
) -> list[MatchIndexRow]:
    merged: dict[str, MatchIndexRow] = {}

    if output_path.exists():
        with output_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                match_id = str(row.get("match_card_id", "")).strip()
                if not match_id:
                    continue
                allowed = {
                    field: row.get(field, "")
                    for field in MatchIndexRow.__dataclass_fields__
                }
                merged[match_id] = MatchIndexRow(**allowed)

    for row in new_rows:
        merged[row.match_card_id] = row

    return sorted(merged.values(), key=lambda row: int(row.match_card_id))


def save_rows(rows: list[MatchIndexRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(MatchIndexRow.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def download_match_pages(
    rows: list[MatchIndexRow],
    force: bool,
    interval: float,
) -> None:
    collector_dir = Path(__file__).resolve().parent
    if str(collector_dir) not in sys.path:
        sys.path.insert(0, str(collector_dir))

    from jleague_collector import JLeagueCollector

    collector = JLeagueCollector(
        request_interval_seconds=interval,
    )
    collector.collect_matches(
        [row.match_card_id for row in rows],
        force=force,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect match_card_id values from the official "
            "J.League Data Site schedule/results search."
        )
    )
    parser.add_argument(
        "--year",
        action="append",
        default=[],
        help=(
            "Official competition_years value. Repeat for multiple years. "
            "Example: --year 2025"
        ),
    )
    parser.add_argument(
        "--competition-frame-id",
        action="append",
        default=[],
        help=(
            "Official competition_frame_ids value. Repeat as needed. "
            "Typical historical league IDs are 1=J1, 2=J2, 3=J3."
        ),
    )
    parser.add_argument(
        "--search-url",
        help=(
            "Use an exact SFMS01/search URL copied from the browser. "
            "When supplied, --year and --competition-frame-id are ignored."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the output instead of merging existing IDs.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download all discovered match detail HTML after indexing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download when --download is used.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between match-detail requests.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    search_url = args.search_url or build_search_url(
        years=args.year,
        competition_frame_ids=args.competition_frame_id,
    )

    LOGGER.info("Search URL: %s", search_url)
    html = fetch_html(search_url, timeout=args.timeout)
    discovered = parse_match_rows(html)

    if not discovered:
        raise RuntimeError(
            "No match_card_id values were found. "
            "Open the official search page in a browser, apply filters, "
            "and pass the resulting URL with --search-url."
        )

    rows = (
        discovered
        if args.replace
        else merge_existing(discovered, args.output)
    )
    save_rows(rows, args.output)

    LOGGER.info("Discovered this run: %s", len(discovered))
    LOGGER.info("Total saved IDs: %s", len(rows))
    LOGGER.info("Saved: %s", args.output)

    if args.download:
        LOGGER.info("Downloading %s match pages...", len(discovered))
        download_match_pages(
            rows=discovered,
            force=args.force,
            interval=args.interval,
        )


if __name__ == "__main__":
    main()
