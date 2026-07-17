from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

from head_to_head.basic_stats import BasicStatsCalculator
from head_to_head.models import (
    MatchOutcome,
    MatchRecord,
    WeightedHeadToHeadStats,
)


class WeightedStatsCalculator:
    """Calculate exponentially time-decayed head-to-head statistics."""

    def __init__(self, half_life_days: float = 730.0) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive.")
        self.half_life_days = float(half_life_days)

    def calculate(
        self,
        matches: Sequence[MatchRecord],
        current_match_date: date,
        current_home_team: str,
        current_away_team: str,
    ) -> WeightedHeadToHeadStats:
        """Aggregate historical matches using exponential time decay."""
        total_weight = 0.0
        weighted_home_wins = 0.0
        weighted_draws = 0.0
        weighted_away_wins = 0.0
        weighted_home_goals = 0.0
        weighted_away_goals = 0.0

        for match in matches:
            if match.match_date >= current_match_date:
                continue

            viewpoint = BasicStatsCalculator.to_current_viewpoint(
                match=match,
                current_home_team=current_home_team,
                current_away_team=current_away_team,
            )
            if viewpoint is None:
                continue

            home_score, away_score, outcome = viewpoint
            days_difference = (current_match_date - match.match_date).days
            weight = self.calculate_weight(days_difference)

            total_weight += weight
            weighted_home_goals += home_score * weight
            weighted_away_goals += away_score * weight

            if outcome is MatchOutcome.HOME_WIN:
                weighted_home_wins += weight
            elif outcome is MatchOutcome.DRAW:
                weighted_draws += weight
            else:
                weighted_away_wins += weight

        return WeightedHeadToHeadStats(
            total_weight=total_weight,
            weighted_home_wins=weighted_home_wins,
            weighted_draws=weighted_draws,
            weighted_away_wins=weighted_away_wins,
            weighted_home_goals=weighted_home_goals,
            weighted_away_goals=weighted_away_goals,
        )

    def calculate_weight(self, days_difference: int) -> float:
        """Return an exponential-decay weight for an age in days."""
        if days_difference < 0:
            raise ValueError("days_difference must not be negative.")

        return math.exp(
            -math.log(2.0) * days_difference / self.half_life_days
        )
