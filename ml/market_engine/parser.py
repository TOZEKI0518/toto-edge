from __future__ import annotations

import logging
import re
from datetime import datetime
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup, Tag

from market_engine.models import MarketMatch

LOGGER = logging.getLogger(__name__)

PERCENT_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
DATE_PATTERN = re.compile(r"\b([0-9]{1,2}/[0-9]{1,2})\b")
MATCH_NUMBER_PATTERN = re.compile(r"^\s*([0-9]{1,2})\s*$")


class TotoMarketParser:
    """Parse official toto market HTML into 13 normalized records."""

    def parse(
        self,
        html: str,
        round_id: int,
        source_url: str,
    ) -> list[MarketMatch]:
        """Parse one official toto round page.

        The official result table represents each match using two adjacent
        outer ``tr`` elements:

        1. match number, date, stadium and team names
        2. home/draw/away vote percentages

        This parser treats those two rows as one match and falls back to
        pandas table parsing only when the direct DOM parser cannot recover
        all 13 matches.
        """
        records = self._parse_official_two_row_table(
            html=html,
            round_id=round_id,
            source_url=source_url,
        )

        if len(records) != 13:
            fallback = self._parse_with_pandas(
                html=html,
                round_id=round_id,
                source_url=source_url,
            )
            records = self._merge_records(records, fallback)

        ordered = [
            records[number]
            for number in sorted(records)
        ]
        self._validate(ordered)
        return ordered

    def _parse_official_two_row_table(
        self,
        html: str,
        round_id: int,
        source_url: str,
    ) -> dict[int, MarketMatch]:
        """Parse the official two-row-per-match HTML structure."""
        soup = BeautifulSoup(html, "html.parser")
        wrapper = soup.select_one("#tableWrap01")
        if wrapper is None:
            return {}

        outer_table = wrapper.find("table", recursive=False)
        if outer_table is None:
            return {}

        outer_rows = outer_table.find_all("tr", recursive=False)
        parsed: dict[int, MarketMatch] = {}

        index = 0
        while index < len(outer_rows):
            information_row = outer_rows[index]
            match_number = self._extract_match_number(information_row)

            if match_number is None:
                index += 1
                continue

            probability_row = (
                outer_rows[index + 1]
                if index + 1 < len(outer_rows)
                else None
            )
            if probability_row is None:
                index += 1
                continue

            home_team, away_team = self._extract_team_pair(
                information_row
            )
            percentages = self._extract_percentages(
                probability_row.get_text(" ", strip=True)
            )

            if len(percentages) < 3:
                LOGGER.warning(
                    "Vote percentages not found for match %d.",
                    match_number,
                )
                index += 2
                continue

            match_day = self._extract_match_day(information_row)

            parsed[match_number] = self._create_record(
                round_id=round_id,
                match_number=match_number,
                home_team=home_team,
                away_team=away_team,
                probabilities=percentages[:3],
                match_day=match_day,
                source_url=source_url,
            )
            index += 2

        return parsed

    @staticmethod
    def _extract_match_number(row: Tag) -> int | None:
        """Extract the 1–13 match number from an outer information row."""
        direct_cells = row.find_all(
            ["td", "th"],
            recursive=False,
        )
        if not direct_cells:
            return None

        text = direct_cells[0].get_text(" ", strip=True)
        match = MATCH_NUMBER_PATTERN.fullmatch(text)
        if match is None:
            return None

        number = int(match.group(1))
        return number if 1 <= number <= 13 else None

    @staticmethod
    def _extract_match_day(row: Tag) -> str:
        """Extract the MM/DD match day from the information row."""
        direct_cells = row.find_all(
            ["td", "th"],
            recursive=False,
        )
        for cell in direct_cells:
            text = cell.get_text(" ", strip=True)
            match = DATE_PATTERN.search(text)
            if match:
                return match.group(1)
        return ""

    def _extract_team_pair(self, row: Tag) -> tuple[str, str]:
        """Extract home and away team names from the nested team table."""
        team_container = row.select_one("td.tableIn01")
        if team_container is None:
            return "", ""

        nested_table = team_container.find("table")
        if nested_table is None:
            return "", ""

        first_row = nested_table.find("tr")
        if first_row is None:
            return "", ""

        cells = first_row.find_all("td", recursive=False)
        values = [
            self._clean_text(cell.get_text(" ", strip=True))
            for cell in cells
        ]

        if len(values) >= 3 and values[1].upper() == "VS":
            return values[0], values[2]

        non_separator_values = [
            value
            for value in values
            if value and value.upper() not in {"VS", "対"}
        ]
        if len(non_separator_values) >= 2:
            return (
                non_separator_values[0],
                non_separator_values[-1],
            )

        return "", ""

    @staticmethod
    def _extract_percentages(text: str) -> list[float]:
        """Extract percentage values as decimal probabilities."""
        return [
            float(value) / 100.0
            for value in PERCENT_PATTERN.findall(text)
        ]

    def _parse_with_pandas(
        self,
        html: str,
        round_id: int,
        source_url: str,
    ) -> dict[int, MarketMatch]:
        """Fallback parser for future minor HTML structure changes."""
        try:
            tables = pd.read_html(StringIO(html))
        except ValueError:
            return {}

        parsed: dict[int, MarketMatch] = {}

        for table in tables:
            frame = self._flatten_columns(table)

            for _, row in frame.iterrows():
                values = [
                    self._clean_text(value)
                    for value in row.tolist()
                ]
                match_number = self._find_number_in_values(values)
                if match_number is None:
                    continue

                combined = " ".join(
                    value for value in values if value
                )
                percentages = self._extract_percentages(combined)
                if len(percentages) < 3:
                    continue

                home_team, away_team = self._find_teams_in_values(
                    values
                )
                date_match = DATE_PATTERN.search(combined)

                parsed[match_number] = self._create_record(
                    round_id=round_id,
                    match_number=match_number,
                    home_team=home_team,
                    away_team=away_team,
                    probabilities=percentages[-3:],
                    match_day=(
                        date_match.group(1)
                        if date_match
                        else ""
                    ),
                    source_url=source_url,
                )

        return parsed

    @staticmethod
    def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        if isinstance(output.columns, pd.MultiIndex):
            output.columns = [
                " ".join(
                    str(part).strip()
                    for part in column
                    if str(part).strip()
                    and str(part).lower() != "nan"
                )
                for column in output.columns
            ]
        return output

    @staticmethod
    def _find_number_in_values(
        values: list[str],
    ) -> int | None:
        for value in values[:4]:
            match = MATCH_NUMBER_PATTERN.fullmatch(value)
            if match:
                number = int(match.group(1))
                if 1 <= number <= 13:
                    return number
        return None

    def _find_teams_in_values(
        self,
        values: list[str],
    ) -> tuple[str, str]:
        """Recover a team pair from flattened table-cell text."""
        for value in values:
            normalized = self._clean_text(value)
            if "VS" not in normalized.upper():
                continue

            parts = re.split(
                r"\s+VS\s+",
                normalized,
                maxsplit=1,
                flags=re.IGNORECASE,
            )
            if len(parts) == 2:
                home = self._remove_metadata(parts[0])
                away = self._remove_metadata(parts[1])
                if home and away:
                    return home, away

        return "", ""

    @staticmethod
    def _remove_metadata(value: str) -> str:
        cleaned = DATE_PATTERN.sub("", value)
        cleaned = PERCENT_PATTERN.sub("", cleaned)
        cleaned = re.sub(
            r"^\s*(?:[0-9]{1,2})\s+",
            "",
            cleaned,
        )
        return re.sub(r"\s+", " ", cleaned).strip(" -　")

    @staticmethod
    def _clean_text(value: object) -> str:
        if pd.isna(value):
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def _create_record(
        self,
        *,
        round_id: int,
        match_number: int,
        home_team: str,
        away_team: str,
        probabilities: list[float],
        match_day: str,
        source_url: str,
    ) -> MarketMatch:
        if len(probabilities) < 3:
            raise ValueError(
                f"Three probabilities are required for match "
                f"{match_number}."
            )

        home_probability = float(probabilities[0])
        draw_probability = float(probabilities[1])
        away_probability = float(probabilities[2])
        total = (
            home_probability
            + draw_probability
            + away_probability
        )

        if total <= 0:
            raise ValueError(
                f"Probability sum is zero for match {match_number}."
            )

        record = MarketMatch(
            round_id=round_id,
            toto_match_no=match_number,
            home_team=self._clean_text(home_team),
            away_team=self._clean_text(away_team),
            market_prob_home=home_probability / total,
            market_prob_draw=draw_probability / total,
            market_prob_away=away_probability / total,
            match_day=match_day,
            collected_at=datetime.now().isoformat(
                timespec="seconds"
            ),
            source_url=source_url,
        )
        return record

    @staticmethod
    def _merge_records(
        primary: dict[int, MarketMatch],
        fallback: dict[int, MarketMatch],
    ) -> dict[int, MarketMatch]:
        """Merge fallback data without replacing valid primary values."""
        merged = dict(primary)

        for number, fallback_record in fallback.items():
            existing = merged.get(number)
            if existing is None:
                merged[number] = fallback_record
                continue

            if (
                not existing.home_team
                and fallback_record.home_team
            ):
                merged[number] = MarketMatch(
                    round_id=existing.round_id,
                    toto_match_no=existing.toto_match_no,
                    home_team=fallback_record.home_team,
                    away_team=fallback_record.away_team,
                    market_prob_home=existing.market_prob_home,
                    market_prob_draw=existing.market_prob_draw,
                    market_prob_away=existing.market_prob_away,
                    match_day=(
                        existing.match_day
                        or fallback_record.match_day
                    ),
                    collected_at=existing.collected_at,
                    source_url=existing.source_url,
                )

        return merged

    @staticmethod
    def _validate(records: list[MarketMatch]) -> None:
        """Validate complete 13-match output."""
        if len(records) != 13:
            raise ValueError(
                f"Expected 13 toto matches, parsed {len(records)}."
            )

        match_numbers = [
            record.toto_match_no
            for record in records
        ]
        if match_numbers != list(range(1, 14)):
            raise ValueError(
                "Expected toto match numbers 1 through 13, "
                f"parsed {match_numbers}."
            )

        missing_teams = [
            record.toto_match_no
            for record in records
            if not record.home_team
            or not record.away_team
        ]
        if missing_teams:
            raise ValueError(
                "Team names could not be parsed for matches: "
                + ", ".join(map(str, missing_teams))
            )

        for record in records:
            probabilities = (
                float(record.market_prob_home),
                float(record.market_prob_draw),
                float(record.market_prob_away),
            )
            if any(value < 0 for value in probabilities):
                raise ValueError(
                    f"Negative market probability found for match "
                    f"{record.toto_match_no}."
                )
            if sum(probabilities) <= 0:
                raise ValueError(
                    f"Probability sum is zero for match "
                    f"{record.toto_match_no}."
                )
