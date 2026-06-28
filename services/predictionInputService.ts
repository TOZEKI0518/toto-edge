import { normalizeTeamName } from "@/lib/teamNameNormalizer";
import type { PredictionInput, TotoFixture } from "@/types/fixture";
import type { TeamStanding } from "@/types/standing";

export function buildPredictionInputs(
  fixtures: TotoFixture[],
  standings: TeamStanding[]
): PredictionInput[] {
  return fixtures.map((fixture) => {
    const homeName = normalizeTeamName(fixture.homeTeam);
    const awayName = normalizeTeamName(fixture.awayTeam);

    const homeStanding = standings.find(
      (team) => normalizeTeamName(team.teamName) === homeName
    );

    const awayStanding = standings.find(
      (team) => normalizeTeamName(team.teamName) === awayName
    );

    return {
      fixture: {
        ...fixture,
        homeTeam: homeName,
        awayTeam: awayName,
      },
      homeStanding,
      awayStanding,
    };
  });
}