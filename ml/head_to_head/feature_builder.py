from __future__ import annotations

from collections.abc import Sequence

from head_to_head.basic_stats import BasicStatsCalculator
from head_to_head.models import (
    HeadToHeadEngineConfig,
    HeadToHeadFeatureRow,
    MatchOutcome,
    MatchRecord,
)
from head_to_head.repository import HeadToHeadRepository
from head_to_head.weighted_stats import WeightedStatsCalculator


class HeadToHeadFeatureBuilder:
    """Build one leakage-safe head-to-head feature row for a target match."""

    def __init__(
        self,
        repository: HeadToHeadRepository,
        config: HeadToHeadEngineConfig,
    ) -> None:
        config.validate()
        self.repository = repository
        self.config = config
        self.basic_calculator = BasicStatsCalculator()
        self.weighted_calculator = WeightedStatsCalculator(
            half_life_days=config.decay_half_life_days
        )

    def build(self, current_match: MatchRecord) -> HeadToHeadFeatureRow:
        """Build all H2H features using only matches before the target date."""
        history = self.repository.get_history(
            home_team=current_match.home_team,
            away_team=current_match.away_team,
            before_date=current_match.match_date,
        )
        history = history[-self.config.max_history_matches :]

        if not history:
            return HeadToHeadFeatureRow.empty(
                match_card_id=current_match.match_card_id,
                default_days_since_last_match=(
                    self.config.default_days_since_last_match
                ),
            )

        full_stats = self.basic_calculator.calculate(
            history,
            current_match.home_team,
            current_match.away_team,
        )

        last3_stats = self.basic_calculator.calculate(
            history[-self.config.recent_window_3 :],
            current_match.home_team,
            current_match.away_team,
        )
        last5_stats = self.basic_calculator.calculate(
            history[-self.config.recent_window_5 :],
            current_match.home_team,
            current_match.away_team,
        )

        same_venue_history = self.repository.get_same_venue_history(
            home_team=current_match.home_team,
            away_team=current_match.away_team,
            before_date=current_match.match_date,
        )
        same_venue_history = same_venue_history[
            -self.config.max_history_matches :
        ]
        same_venue_stats = self.basic_calculator.calculate(
            same_venue_history,
            current_match.home_team,
            current_match.away_team,
        )

        weighted_stats = self.weighted_calculator.calculate(
            matches=history,
            current_match_date=current_match.match_date,
            current_home_team=current_match.home_team,
            current_away_team=current_match.away_team,
        )

        latest_match = history[-1]
        days_since_last_match = self.repository.days_since(
            current_date=current_match.match_date,
            previous_date=latest_match.match_date,
        )

        home_unbeaten_streak, away_unbeaten_streak = (
            self._calculate_unbeaten_streaks(
                matches=history,
                current_home_team=current_match.home_team,
                current_away_team=current_match.away_team,
            )
        )

        confidence = min(
            full_stats.matches
            / self.config.confidence_saturation_matches,
            1.0,
        )

        return HeadToHeadFeatureRow(
            match_card_id=current_match.match_card_id,
            h2h_matches=full_stats.matches,
            h2h_home_win_rate=full_stats.home_win_rate,
            h2h_draw_rate=full_stats.draw_rate,
            h2h_away_win_rate=full_stats.away_win_rate,
            h2h_home_points_per_game=full_stats.home_points_per_game,
            h2h_away_points_per_game=full_stats.away_points_per_game,
            h2h_average_home_goals=full_stats.average_home_goals,
            h2h_average_away_goals=full_stats.average_away_goals,
            h2h_average_goal_difference=full_stats.average_goal_difference,
            h2h_btts_rate=full_stats.btts_rate,
            h2h_over_2_5_rate=full_stats.over_2_5_rate,
            h2h_last3_home_win_rate=last3_stats.home_win_rate,
            h2h_last3_draw_rate=last3_stats.draw_rate,
            h2h_last3_away_win_rate=last3_stats.away_win_rate,
            h2h_last3_goal_difference=last3_stats.average_goal_difference,
            h2h_last5_home_win_rate=last5_stats.home_win_rate,
            h2h_last5_draw_rate=last5_stats.draw_rate,
            h2h_last5_away_win_rate=last5_stats.away_win_rate,
            h2h_last5_goal_difference=last5_stats.average_goal_difference,
            h2h_same_venue_matches=same_venue_stats.matches,
            h2h_same_venue_home_win_rate=same_venue_stats.home_win_rate,
            h2h_same_venue_draw_rate=same_venue_stats.draw_rate,
            h2h_same_venue_away_win_rate=same_venue_stats.away_win_rate,
            h2h_same_venue_goal_difference=(
                same_venue_stats.average_goal_difference
            ),
            h2h_weighted_home_win_rate=weighted_stats.home_win_rate,
            h2h_weighted_draw_rate=weighted_stats.draw_rate,
            h2h_weighted_away_win_rate=weighted_stats.away_win_rate,
            h2h_weighted_goal_difference=(
                weighted_stats.average_goal_difference
            ),
            h2h_days_since_last_match=days_since_last_match,
            h2h_home_unbeaten_streak=home_unbeaten_streak,
            h2h_away_unbeaten_streak=away_unbeaten_streak,
            h2h_confidence=confidence,
        )

    def _calculate_unbeaten_streaks(
        self,
        matches: Sequence[MatchRecord],
        current_home_team: str,
        current_away_team: str,
    ) -> tuple[int, int]:
        """Return consecutive unbeaten runs from newest historical match."""
        home_streak = 0
        away_streak = 0
        home_active = True
        away_active = True

        for match in reversed(matches):
            viewpoint = self.basic_calculator.to_current_viewpoint(
                match=match,
                current_home_team=current_home_team,
                current_away_team=current_away_team,
            )
            if viewpoint is None:
                continue

            _, _, outcome = viewpoint

            if home_active:
                if outcome in (MatchOutcome.HOME_WIN, MatchOutcome.DRAW):
                    home_streak += 1
                else:
                    home_active = False

            if away_active:
                if outcome in (MatchOutcome.AWAY_WIN, MatchOutcome.DRAW):
                    away_streak += 1
                else:
                    away_active = False

            if not home_active and not away_active:
                break

        return home_streak, away_streak
