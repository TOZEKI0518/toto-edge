from __future__ import annotations

from collections.abc import Sequence

from head_to_head.models import (
    BasicHeadToHeadStats,
    MatchOutcome,
    MatchRecord,
)


class BasicStatsCalculator:
    """Calculate unweighted head-to-head statistics from the current match viewpoint."""

    def calculate(
        self,
        matches: Sequence[MatchRecord],
        current_home_team: str,
        current_away_team: str,
    ) -> BasicHeadToHeadStats:
        """Aggregate historical matches from the current home/away viewpoint.

        Historical fixtures may have been played with the venue reversed. Every
        record is therefore transformed so that goals and outcomes are expressed
        from ``current_home_team`` versus ``current_away_team``.
        """
        if current_home_team == current_away_team:
            raise ValueError("current_home_team and current_away_team must differ.")

        home_wins = 0
        draws = 0
        away_wins = 0
        home_goals = 0.0
        away_goals = 0.0
        both_teams_scored_matches = 0
        over_2_5_matches = 0

        for match in matches:
            viewpoint = self.to_current_viewpoint(
                match=match,
                current_home_team=current_home_team,
                current_away_team=current_away_team,
            )

            if viewpoint is None:
                continue

            home_score, away_score, outcome = viewpoint
            home_goals += home_score
            away_goals += away_score

            if outcome is MatchOutcome.HOME_WIN:
                home_wins += 1
            elif outcome is MatchOutcome.DRAW:
                draws += 1
            else:
                away_wins += 1

            if home_score > 0 and away_score > 0:
                both_teams_scored_matches += 1
            if home_score + away_score > 2.5:
                over_2_5_matches += 1

        matches_count = home_wins + draws + away_wins

        return BasicHeadToHeadStats(
            matches=matches_count,
            home_wins=home_wins,
            draws=draws,
            away_wins=away_wins,
            home_goals=home_goals,
            away_goals=away_goals,
            both_teams_scored_matches=both_teams_scored_matches,
            over_2_5_matches=over_2_5_matches,
        )

    @staticmethod
    def to_current_viewpoint(
        match: MatchRecord,
        current_home_team: str,
        current_away_team: str,
    ) -> tuple[float, float, MatchOutcome] | None:
        """Transform one historical match into the current fixture viewpoint."""
        same_direction = (
            match.home_team == current_home_team
            and match.away_team == current_away_team
        )
        reversed_direction = (
            match.home_team == current_away_team
            and match.away_team == current_home_team
        )

        if same_direction:
            home_score = float(match.home_score)
            away_score = float(match.away_score)
        elif reversed_direction:
            home_score = float(match.away_score)
            away_score = float(match.home_score)
        else:
            return None

        outcome = MatchOutcome.from_scores(home_score, away_score)
        if outcome is None:
            return None

        return home_score, away_score, outcome
