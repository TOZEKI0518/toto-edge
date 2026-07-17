from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Final


class MatchOutcome(str, Enum):
    HOME_WIN = "H"
    DRAW = "D"
    AWAY_WIN = "A"

    @classmethod
    def from_scores(
        cls,
        home_score: int | float | None,
        away_score: int | float | None,
    ) -> "MatchOutcome | None":
        if home_score is None or away_score is None:
            return None
        if home_score > away_score:
            return cls.HOME_WIN
        if home_score < away_score:
            return cls.AWAY_WIN
        return cls.DRAW


@dataclass(frozen=True, slots=True)
class TeamPair:
    home_team: str
    away_team: str

    def reversed(self) -> "TeamPair":
        return TeamPair(self.away_team, self.home_team)

    @property
    def unordered_key(self) -> tuple[str, str]:
        return tuple(sorted((self.home_team, self.away_team)))


@dataclass(frozen=True, slots=True)
class MatchRecord:
    match_card_id: int
    match_date: date
    season: int
    competition: str
    home_team: str
    away_team: str
    home_score: float
    away_score: float
    outcome: MatchOutcome

    @property
    def pair(self) -> TeamPair:
        return TeamPair(self.home_team, self.away_team)

    @property
    def total_goals(self) -> float:
        return self.home_score + self.away_score

    @property
    def goal_difference(self) -> float:
        return self.home_score - self.away_score

    @property
    def both_teams_scored(self) -> bool:
        return self.home_score > 0 and self.away_score > 0

    @property
    def over_2_5(self) -> bool:
        return self.total_goals > 2.5


@dataclass(frozen=True, slots=True)
class WeightedMatch:
    match: MatchRecord
    weight: float


@dataclass(frozen=True, slots=True)
class BasicHeadToHeadStats:
    matches: int = 0
    home_wins: int = 0
    draws: int = 0
    away_wins: int = 0
    home_goals: float = 0.0
    away_goals: float = 0.0
    both_teams_scored_matches: int = 0
    over_2_5_matches: int = 0

    @property
    def home_win_rate(self) -> float:
        return self.home_wins / self.matches if self.matches else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.matches if self.matches else 0.0

    @property
    def away_win_rate(self) -> float:
        return self.away_wins / self.matches if self.matches else 0.0

    @property
    def home_points_per_game(self) -> float:
        return (
            (self.home_wins * 3 + self.draws) / self.matches
            if self.matches
            else 0.0
        )

    @property
    def away_points_per_game(self) -> float:
        return (
            (self.away_wins * 3 + self.draws) / self.matches
            if self.matches
            else 0.0
        )

    @property
    def average_home_goals(self) -> float:
        return self.home_goals / self.matches if self.matches else 0.0

    @property
    def average_away_goals(self) -> float:
        return self.away_goals / self.matches if self.matches else 0.0

    @property
    def average_goal_difference(self) -> float:
        return (
            (self.home_goals - self.away_goals) / self.matches
            if self.matches
            else 0.0
        )

    @property
    def btts_rate(self) -> float:
        return (
            self.both_teams_scored_matches / self.matches
            if self.matches
            else 0.0
        )

    @property
    def over_2_5_rate(self) -> float:
        return self.over_2_5_matches / self.matches if self.matches else 0.0


@dataclass(frozen=True, slots=True)
class WeightedHeadToHeadStats:
    total_weight: float = 0.0
    weighted_home_wins: float = 0.0
    weighted_draws: float = 0.0
    weighted_away_wins: float = 0.0
    weighted_home_goals: float = 0.0
    weighted_away_goals: float = 0.0

    @property
    def home_win_rate(self) -> float:
        return (
            self.weighted_home_wins / self.total_weight
            if self.total_weight > 0
            else 0.0
        )

    @property
    def draw_rate(self) -> float:
        return (
            self.weighted_draws / self.total_weight
            if self.total_weight > 0
            else 0.0
        )

    @property
    def away_win_rate(self) -> float:
        return (
            self.weighted_away_wins / self.total_weight
            if self.total_weight > 0
            else 0.0
        )

    @property
    def average_goal_difference(self) -> float:
        return (
            (
                self.weighted_home_goals
                - self.weighted_away_goals
            )
            / self.total_weight
            if self.total_weight > 0
            else 0.0
        )


@dataclass(frozen=True, slots=True)
class HeadToHeadFeatureRow:
    match_card_id: int
    h2h_matches: int
    h2h_home_win_rate: float
    h2h_draw_rate: float
    h2h_away_win_rate: float
    h2h_home_points_per_game: float
    h2h_away_points_per_game: float
    h2h_average_home_goals: float
    h2h_average_away_goals: float
    h2h_average_goal_difference: float
    h2h_btts_rate: float
    h2h_over_2_5_rate: float
    h2h_last3_home_win_rate: float
    h2h_last3_draw_rate: float
    h2h_last3_away_win_rate: float
    h2h_last3_goal_difference: float
    h2h_last5_home_win_rate: float
    h2h_last5_draw_rate: float
    h2h_last5_away_win_rate: float
    h2h_last5_goal_difference: float
    h2h_same_venue_matches: int
    h2h_same_venue_home_win_rate: float
    h2h_same_venue_draw_rate: float
    h2h_same_venue_away_win_rate: float
    h2h_same_venue_goal_difference: float
    h2h_weighted_home_win_rate: float
    h2h_weighted_draw_rate: float
    h2h_weighted_away_win_rate: float
    h2h_weighted_goal_difference: float
    h2h_days_since_last_match: int
    h2h_home_unbeaten_streak: int
    h2h_away_unbeaten_streak: int
    h2h_confidence: float

    @classmethod
    def empty(
        cls,
        match_card_id: int,
        default_days_since_last_match: int,
    ) -> "HeadToHeadFeatureRow":
        return cls(
            match_card_id=match_card_id,
            h2h_matches=0,
            h2h_home_win_rate=0.0,
            h2h_draw_rate=0.0,
            h2h_away_win_rate=0.0,
            h2h_home_points_per_game=0.0,
            h2h_away_points_per_game=0.0,
            h2h_average_home_goals=0.0,
            h2h_average_away_goals=0.0,
            h2h_average_goal_difference=0.0,
            h2h_btts_rate=0.0,
            h2h_over_2_5_rate=0.0,
            h2h_last3_home_win_rate=0.0,
            h2h_last3_draw_rate=0.0,
            h2h_last3_away_win_rate=0.0,
            h2h_last3_goal_difference=0.0,
            h2h_last5_home_win_rate=0.0,
            h2h_last5_draw_rate=0.0,
            h2h_last5_away_win_rate=0.0,
            h2h_last5_goal_difference=0.0,
            h2h_same_venue_matches=0,
            h2h_same_venue_home_win_rate=0.0,
            h2h_same_venue_draw_rate=0.0,
            h2h_same_venue_away_win_rate=0.0,
            h2h_same_venue_goal_difference=0.0,
            h2h_weighted_home_win_rate=0.0,
            h2h_weighted_draw_rate=0.0,
            h2h_weighted_away_win_rate=0.0,
            h2h_weighted_goal_difference=0.0,
            h2h_days_since_last_match=default_days_since_last_match,
            h2h_home_unbeaten_streak=0,
            h2h_away_unbeaten_streak=0,
            h2h_confidence=0.0,
        )


@dataclass(frozen=True, slots=True)
class HeadToHeadEngineConfig:
    input_matches_csv: Path
    output_features_csv: Path
    diagnostics_dir: Path
    decay_half_life_days: float = 730.0
    max_history_matches: int = 20
    recent_window_3: int = 3
    recent_window_5: int = 5
    default_days_since_last_match: int = 3650
    confidence_saturation_matches: int = 8

    def validate(self) -> None:
        if self.decay_half_life_days <= 0:
            raise ValueError("decay_half_life_days must be positive.")
        if self.max_history_matches <= 0:
            raise ValueError("max_history_matches must be positive.")
        if self.recent_window_3 <= 0:
            raise ValueError("recent_window_3 must be positive.")
        if self.recent_window_5 <= 0:
            raise ValueError("recent_window_5 must be positive.")
        if self.recent_window_3 > self.recent_window_5:
            raise ValueError(
                "recent_window_3 must not exceed recent_window_5."
            )
        if self.confidence_saturation_matches <= 0:
            raise ValueError(
                "confidence_saturation_matches must be positive."
            )


DEFAULT_OUTPUT_COLUMNS: Final[tuple[str, ...]] = tuple(
    HeadToHeadFeatureRow.__dataclass_fields__.keys()
)
