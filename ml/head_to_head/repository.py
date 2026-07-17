from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from head_to_head.models import MatchOutcome, MatchRecord

LOGGER = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {
    "match_card_id",
    "match_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}


@dataclass(frozen=True, slots=True)
class RepositoryConfig:
    """Repository input configuration."""

    matches_csv: Path

    def validate(self) -> None:
        if not self.matches_csv.exists():
            raise FileNotFoundError(
                f"matches CSV was not found: {self.matches_csv}"
            )
        if not self.matches_csv.is_file():
            raise ValueError(
                f"matches_csv is not a file: {self.matches_csv}"
            )


class HeadToHeadRepository:
    """Leakage-safe repository for chronological head-to-head searches."""

    def __init__(self, config: RepositoryConfig) -> None:
        config.validate()
        self.config = config
        self.matches: list[MatchRecord] = []
        self.by_pair: dict[tuple[str, str], list[MatchRecord]] = defaultdict(list)
        self.by_home_pair: dict[
            tuple[str, str], list[MatchRecord]
        ] = defaultdict(list)
        self.by_team: dict[str, list[MatchRecord]] = defaultdict(list)

    def load(self) -> None:
        """Load valid completed matches and build lookup indices."""
        dataframe = pd.read_csv(self.config.matches_csv)
        missing_columns = sorted(_REQUIRED_COLUMNS - set(dataframe.columns))
        if missing_columns:
            raise ValueError(
                "matches CSV is missing required columns: "
                + ", ".join(missing_columns)
            )

        dataframe["match_date"] = pd.to_datetime(
            dataframe["match_date"],
            errors="coerce",
        )
        invalid_dates = int(dataframe["match_date"].isna().sum())
        if invalid_dates:
            raise ValueError(
                f"matches CSV contains {invalid_dates} invalid match_date values."
            )

        duplicate_ids = int(dataframe["match_card_id"].duplicated().sum())
        if duplicate_ids:
            raise ValueError(
                f"matches CSV contains {duplicate_ids} duplicate match_card_id values."
            )

        dataframe = dataframe.sort_values(
            ["match_date", "match_card_id"],
            kind="stable",
        )

        self.matches.clear()
        self.by_pair.clear()
        self.by_home_pair.clear()
        self.by_team.clear()

        skipped = 0
        for row in dataframe.itertuples(index=False):
            outcome = MatchOutcome.from_scores(
                getattr(row, "home_score"),
                getattr(row, "away_score"),
            )
            if outcome is None:
                skipped += 1
                continue

            match_date = getattr(row, "match_date").date()
            season_value = getattr(row, "season", match_date.year)
            competition_value = getattr(row, "competition", "J.League")

            match = MatchRecord(
                match_card_id=int(getattr(row, "match_card_id")),
                match_date=match_date,
                season=int(season_value),
                competition=str(competition_value),
                home_team=str(getattr(row, "home_team")).strip(),
                away_team=str(getattr(row, "away_team")).strip(),
                home_score=float(getattr(row, "home_score")),
                away_score=float(getattr(row, "away_score")),
                outcome=outcome,
            )

            if not match.home_team or not match.away_team:
                skipped += 1
                continue

            self.matches.append(match)
            self.by_pair[match.pair.unordered_key].append(match)
            self.by_home_pair[
                (match.home_team, match.away_team)
            ].append(match)
            self.by_team[match.home_team].append(match)
            self.by_team[match.away_team].append(match)

        LOGGER.info(
            "Loaded %d matches from %s; skipped=%d",
            len(self.matches),
            self.config.matches_csv,
            skipped,
        )

    def get_history(
        self,
        home_team: str,
        away_team: str,
        before_date: date,
    ) -> list[MatchRecord]:
        key = tuple(sorted((home_team, away_team)))
        return [
            match
            for match in self.by_pair.get(key, [])
            if match.match_date < before_date
        ]

    def get_same_venue_history(
        self,
        home_team: str,
        away_team: str,
        before_date: date,
    ) -> list[MatchRecord]:
        return [
            match
            for match in self.by_home_pair.get(
                (home_team, away_team), []
            )
            if match.match_date < before_date
        ]

    def get_last_matches(
        self,
        home_team: str,
        away_team: str,
        before_date: date,
        n: int,
    ) -> list[MatchRecord]:
        if n <= 0:
            return []
        return self.get_history(home_team, away_team, before_date)[-n:]

    def get_team_history(
        self,
        team: str,
        before_date: date,
    ) -> list[MatchRecord]:
        return [
            match
            for match in self.by_team.get(team, [])
            if match.match_date < before_date
        ]

    def get_last_team_matches(
        self,
        team: str,
        before_date: date,
        n: int,
    ) -> list[MatchRecord]:
        if n <= 0:
            return []
        return self.get_team_history(team, before_date)[-n:]

    @staticmethod
    def days_since(current_date: date, previous_date: date) -> int:
        days = (current_date - previous_date).days
        if days < 0:
            raise ValueError("previous_date must not be after current_date.")
        return days

    def latest_match(
        self,
        home_team: str,
        away_team: str,
        before_date: date,
    ) -> MatchRecord | None:
        history = self.get_history(home_team, away_team, before_date)
        return history[-1] if history else None

    def pair_exists(self, home_team: str, away_team: str) -> bool:
        return tuple(sorted((home_team, away_team))) in self.by_pair
