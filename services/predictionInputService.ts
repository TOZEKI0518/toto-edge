import { normalizeTeamName } from "@/lib/teamNameNormalizer";
import type { PredictionInput, TotoFixture } from "@/types/fixture";
import type { TeamStanding } from "@/types/standing";

function findStanding(teamName: string, standings: TeamStanding[]) {
  const normalized = normalizeTeamName(teamName);

  return standings.find(
    (standing) => normalizeTeamName(standing.teamName) === normalized
  );
}

export function buildPredictionInputs(
  fixtures: TotoFixture[],
  standings: TeamStanding[]
): PredictionInput[] {
  return fixtures.map((fixture) => {
    const homeTeam = normalizeTeamName(fixture.homeTeam);
    const awayTeam = normalizeTeamName(fixture.awayTeam);

    return {
      fixture: {
        ...fixture,
        homeTeam,
        awayTeam,
      },
      homeStanding: findStanding(homeTeam, standings),
      awayStanding: findStanding(awayTeam, standings),
    };
  });
}
