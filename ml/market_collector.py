from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger("market_collector")

OFFICIAL_BASE_URL = (
    "https://sp.toto-dream.com/dcs/subos/screen/si01/ssin025/"
    "PGSSIN02501ForwardVotetotoSP.form"
)
ROUND_DISCOVERY_URL = (
    "https://store.toto-dream.com/dcs/subos/screen/ps01/spsl000/"
    "PGSPSL00001InitTotoMulti.form"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/150.0 Safari/537.36"
)

PERCENT_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
ROUND_PATTERN = re.compile(
    r"(?:第\s*)?([0-9]{3,5})\s*回"
)
TEAM_VS_PATTERN = re.compile(
    r"(?P<home>[^ \t\r\n]+(?:[ \t][^ \t\r\n]+)*)"
    r"\s*(?:VS|対)\s*"
    r"(?P<away>[^ \t\r\n]+(?:[ \t][^ \t\r\n]+)*)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    """Configuration for official toto market-data collection."""

    output_dir: Path
    round_id: int | None = None
    timeout_seconds: float = 30.0
    retries: int = 3
    retry_wait_seconds: float = 2.0
    verify_ssl: bool = True
    save_html: bool = True

    def validate(self) -> None:
        if self.round_id is not None and self.round_id <= 0:
            raise ValueError("round_id must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.retries <= 0:
            raise ValueError("retries must be positive.")
        if self.retry_wait_seconds < 0:
            raise ValueError("retry_wait_seconds must not be negative.")


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    """Machine-readable market collection summary."""

    round_id: int
    rows: int
    probability_columns: int
    probability_sum_min: float
    probability_sum_max: float
    duplicate_match_numbers: int
    missing_probability_values: int
    source_url: str
    collected_at: str
    output_csv: str


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,*/*;q=0.8"
            ),
        }
    )
    return session


def request_text(
    session: requests.Session,
    url: str,
    config: CollectorConfig,
) -> str:
    """Fetch a page with bounded retries and encoding normalization."""
    last_error: Exception | None = None

    for attempt in range(1, config.retries + 1):
        try:
            response = session.get(
                url,
                timeout=config.timeout_seconds,
                verify=config.verify_ssl,
            )
            response.raise_for_status()
            if response.apparent_encoding:
                response.encoding = response.apparent_encoding

            text = response.text
            if not text.strip():
                raise ValueError("Official page returned an empty response.")

            LOGGER.info(
                "Fetched official page: status=%d bytes=%d attempt=%d",
                response.status_code,
                len(response.content),
                attempt,
            )
            return text
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            LOGGER.warning(
                "Fetch attempt %d/%d failed: %s",
                attempt,
                config.retries,
                exc,
            )
            if attempt < config.retries:
                time.sleep(config.retry_wait_seconds)

    raise RuntimeError(
        f"Failed to fetch official toto page after {config.retries} attempts."
    ) from last_error


def discover_round_id(
    session: requests.Session,
    config: CollectorConfig,
) -> int:
    """Discover the most recent toto round ID from the official sales page."""
    html = request_text(
        session=session,
        url=ROUND_DISCOVERY_URL,
        config=config,
    )

    candidate_ids: set[int] = set()

    for match in re.finditer(r"holdCntId(?:=|%3D)([0-9]{3,5})", html):
        candidate_ids.add(int(match.group(1)))

    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    for match in ROUND_PATTERN.finditer(text):
        candidate_ids.add(int(match.group(1)))

    if not candidate_ids:
        raise ValueError(
            "Could not discover a toto round ID from the official sales page. "
            "Run again with --round-id."
        )

    round_id = max(candidate_ids)
    LOGGER.info("Discovered latest candidate round ID: %d", round_id)
    return round_id


def build_market_url(round_id: int) -> str:
    query = urlencode(
        {
            "commodityId": "01",
            "fromId": "SSIN026",
            "gameAssortment": "9",
            "holdCntId": str(round_id),
        }
    )
    return f"{OFFICIAL_BASE_URL}?{query}"


def clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if isinstance(output.columns, pd.MultiIndex):
        output.columns = [
            " ".join(
                clean_cell(part)
                for part in column
                if clean_cell(part)
            )
            for column in output.columns
        ]
    else:
        output.columns = [clean_cell(column) for column in output.columns]
    return output


def extract_percentages(text: str) -> list[float]:
    return [
        float(value) / 100.0
        for value in PERCENT_PATTERN.findall(text)
    ]


def find_match_number(row_values: list[str]) -> int | None:
    for value in row_values[:3]:
        match = re.fullmatch(r"\s*([0-9]{1,2})\s*", value)
        if match:
            number = int(match.group(1))
            if 1 <= number <= 13:
                return number
    return None


def parse_teams_from_text(text: str) -> tuple[str, str] | None:
    normalized = re.sub(r"\s+", " ", text).strip()
    match = TEAM_VS_PATTERN.search(normalized)
    if not match:
        return None

    home = match.group("home").strip(" -　")
    away = match.group("away").strip(" -　")
    if not home or not away:
        return None
    return home, away


def parse_table_candidates(html: str) -> list[pd.DataFrame]:
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return []
    return [flatten_columns(table) for table in tables]


def parse_market_table(html: str, round_id: int) -> pd.DataFrame:
    """Parse 13 toto matches and 1/D/2 public-pick probabilities."""
    parsed_rows: dict[int, dict[str, object]] = {}

    for table in parse_table_candidates(html):
        for _, row in table.iterrows():
            cells = [clean_cell(value) for value in row.tolist()]
            row_text = " ".join(cell for cell in cells if cell)

            match_number = find_match_number(cells)
            if match_number is None:
                continue

            percentages = extract_percentages(row_text)
            if len(percentages) < 3:
                continue

            teams = parse_teams_from_text(row_text)
            home_team = teams[0] if teams else ""
            away_team = teams[1] if teams else ""

            date_match = re.search(
                r"\b([0-9]{1,2}/[0-9]{1,2})\b",
                row_text,
            )
            match_day = date_match.group(1) if date_match else ""

            parsed_rows[match_number] = {
                "round_id": round_id,
                "toto_match_no": match_number,
                "match_day": match_day,
                "home_team": home_team,
                "away_team": away_team,
                "market_prob_home": percentages[-3],
                "market_prob_draw": percentages[-2],
                "market_prob_away": percentages[-1],
            }

    if len(parsed_rows) != 13:
        fallback = parse_market_from_text(html, round_id)
        for _, row in fallback.iterrows():
            number = int(row["toto_match_no"])
            parsed_rows.setdefault(number, row.to_dict())

    frame = pd.DataFrame(
        [parsed_rows[number] for number in sorted(parsed_rows)]
    )

    if len(frame) != 13:
        raise ValueError(
            "Official page parsing did not produce exactly 13 toto matches. "
            f"Parsed rows: {len(frame)}. The page structure may have changed, "
            "or the specified round may not be a standard toto round."
        )

    return validate_market_frame(frame)


def parse_market_from_text(html: str, round_id: int) -> pd.DataFrame:
    """Fallback parser using visible page text and percentage sequences."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    rows: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        if not re.fullmatch(r"[0-9]{1,2}", line):
            continue

        match_number = int(line)
        if not 1 <= match_number <= 13:
            continue

        block = " ".join(lines[index : index + 18])
        percentages = extract_percentages(block)
        if len(percentages) < 3:
            continue

        teams = parse_teams_from_text(block)
        date_match = re.search(
            r"\b([0-9]{1,2}/[0-9]{1,2})\b",
            block,
        )

        rows.append(
            {
                "round_id": round_id,
                "toto_match_no": match_number,
                "match_day": (
                    date_match.group(1) if date_match else ""
                ),
                "home_team": teams[0] if teams else "",
                "away_team": teams[1] if teams else "",
                "market_prob_home": percentages[0],
                "market_prob_draw": percentages[1],
                "market_prob_away": percentages[2],
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "round_id",
                "toto_match_no",
                "match_day",
                "home_team",
                "away_team",
                "market_prob_home",
                "market_prob_draw",
                "market_prob_away",
            ]
        )

    return pd.DataFrame(rows).drop_duplicates(
        subset=["toto_match_no"],
        keep="first",
    )


def validate_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "round_id",
        "toto_match_no",
        "home_team",
        "away_team",
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Parsed market frame is missing columns: "
            + ", ".join(missing)
        )

    output = frame.copy()
    output = output.sort_values("toto_match_no").reset_index(drop=True)

    if output["toto_match_no"].duplicated().any():
        raise ValueError("Parsed market data contains duplicate match numbers.")

    expected = list(range(1, 14))
    actual = output["toto_match_no"].astype(int).tolist()
    if actual != expected:
        raise ValueError(
            f"Expected match numbers 1..13 but parsed: {actual}"
        )

    probability_columns = [
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
    ]
    output[probability_columns] = output[
        probability_columns
    ].apply(pd.to_numeric, errors="coerce")

    if output[probability_columns].isna().any().any():
        raise ValueError("Parsed market probabilities contain missing values.")
    if (output[probability_columns] < 0).any().any():
        raise ValueError("Parsed market probabilities contain negatives.")

    totals = output[probability_columns].sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Parsed probability row sums to zero.")

    output[probability_columns] = output[
        probability_columns
    ].div(totals, axis=0)

    return output


def create_optimizer_market_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Create optimizer-compatible market data without inventing match IDs."""
    output = frame[
        [
            "round_id",
            "toto_match_no",
            "home_team",
            "away_team",
            "market_prob_away",
            "market_prob_draw",
            "market_prob_home",
        ]
    ].copy()
    return output


def save_outputs(
    frame: pd.DataFrame,
    html: str,
    source_url: str,
    config: CollectorConfig,
    round_id: int,
) -> CollectionSummary:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = (
        config.output_dir
        / f"toto_market_round_{round_id}.csv"
    )
    frame.to_csv(
        detailed_path,
        index=False,
        encoding="utf-8-sig",
    )

    current_path = config.output_dir / "current_toto_market.csv"
    create_optimizer_market_csv(frame).to_csv(
        current_path,
        index=False,
        encoding="utf-8-sig",
    )

    if config.save_html:
        (
            config.output_dir
            / f"toto_market_round_{round_id}.html"
        ).write_text(html, encoding="utf-8")

    probability_columns = [
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
    ]
    sums = frame[probability_columns].sum(axis=1)

    summary = CollectionSummary(
        round_id=round_id,
        rows=len(frame),
        probability_columns=len(probability_columns),
        probability_sum_min=float(sums.min()),
        probability_sum_max=float(sums.max()),
        duplicate_match_numbers=int(
            frame["toto_match_no"].duplicated().sum()
        ),
        missing_probability_values=int(
            frame[probability_columns].isna().sum().sum()
        ),
        source_url=source_url,
        collected_at=datetime.now().isoformat(),
        output_csv=str(detailed_path),
    )

    (
        config.output_dir
        / f"toto_market_round_{round_id}_summary.json"
    ).write_text(
        json.dumps(
            asdict(summary),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return summary


def print_summary(
    summary: CollectionSummary,
    frame: pd.DataFrame,
    config: CollectorConfig,
) -> None:
    print("=" * 92)
    print("Project Alpha Toto Market Collector")
    print("=" * 92)
    print(f"Round ID                  : {summary.round_id}")
    print(f"Rows                      : {summary.rows}")
    print(
        f"Probability sum range     : "
        f"{summary.probability_sum_min:.6f} - "
        f"{summary.probability_sum_max:.6f}"
    )
    print(
        f"Duplicate match numbers   : "
        f"{summary.duplicate_match_numbers}"
    )
    print(
        f"Missing probabilities     : "
        f"{summary.missing_probability_values}"
    )
    print(f"Source URL                : {summary.source_url}")
    print(f"Outputs                   : {config.output_dir}")
    print()
    print(
        frame[
            [
                "toto_match_no",
                "home_team",
                "away_team",
                "market_prob_home",
                "market_prob_draw",
                "market_prob_away",
            ]
        ].to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Collect official toto 13-match public-pick probabilities."
        )
    )
    parser.add_argument(
        "--round-id",
        type=int,
        default=None,
        help=(
            "Official toto round ID. If omitted, attempts automatic discovery."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "ml" / "market_data",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--no-save-html",
        action="store_true",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Use only when the local certificate environment requires it.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    config = CollectorConfig(
        output_dir=args.output_dir,
        round_id=args.round_id,
        timeout_seconds=args.timeout,
        retries=args.retries,
        verify_ssl=not args.no_verify_ssl,
        save_html=not args.no_save_html,
    )
    config.validate()

    session = build_session()
    round_id = (
        config.round_id
        if config.round_id is not None
        else discover_round_id(session, config)
    )
    source_url = build_market_url(round_id)
    html = request_text(
        session=session,
        url=source_url,
        config=config,
    )
    frame = parse_market_table(
        html=html,
        round_id=round_id,
    )
    summary = save_outputs(
        frame=frame,
        html=html,
        source_url=source_url,
        config=config,
        round_id=round_id,
    )
    print_summary(summary, frame, config)


if __name__ == "__main__":
    main()
