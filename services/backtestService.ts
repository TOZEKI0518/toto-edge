import { buildPredictionInputs } from "@/services/predictionInputService";
import { predictFixture } from "@/services/fixturePredictionService";
import { getJleagueStandings } from "@/services/jleagueStandingService";
import { fetchHtml } from "@/services/totoFetcher";
import { parseTotoFixturesFromHtml } from "@/services/totoFixtureParser";
import { mapTotoResultToOutcome } from "@/services/backtestOutcomeMapper";
import type { BacktestRoundResult } from "@/types/backtest";

function getRoundDetailUrl(roundNumber: string) {
  return `https://store.toto-dream.com/dcs/subos/screen/pi04/spin011/PGSPIN01101LnkHoldCntLotResultLsttoto.form?popupDispDiv=disp&holdCntId=${roundNumber}`;
}

export async function runBacktestForRound(
  roundNumber: string
): Promise<BacktestRoundResult> {
  const [standings, html] = await Promise.all([
    getJleagueStandings(),
    fetchHtml(getRoundDetailUrl(roundNumber)),
  ]);

  const fixtures = parseTotoFixturesFromHtml(html);
  const inputs = buildPredictionInputs(fixtures, standings);

  const validInputs = inputs.filter(
    (input) => input.homeStanding && input.awayStanding
  );

  const matches = validInputs
    .map((input) => {
      const prediction = predictFixture(input);
      const fixture = fixtures.find(
        (item) => item.matchNo === prediction.matchNo
      );

      const actualOutcome = mapTotoResultToOutcome(fixture?.totoResult ?? "");

      if (!actualOutcome) return null;

      return {
        matchNo: prediction.matchNo,
        homeTeam: prediction.homeTeam,
        awayTeam: prediction.awayTeam,
        predictedOutcome: prediction.outcome,
        actualOutcome,
        probability: prediction.probability,
        confidence: prediction.confidence,
        totalScore: prediction.totalScore,
        isHit: prediction.outcome === actualOutcome,
      };
    })
    .filter((match): match is NonNullable<typeof match> => match !== null);

  const hitCount = matches.filter((match) => match.isHit).length;
  const totalMatches = matches.length;

  return {
    round: `第${roundNumber}回`,
    totalMatches,
    hitCount,
    hitRate:
      totalMatches === 0 ? 0 : Math.round((hitCount / totalMatches) * 1000) / 10,
    matches,
  };
}
export async function runBacktestSummary(roundNumbers: string[]) {
  const results = await Promise.all(
    roundNumbers.map((round) => runBacktestForRound(round))
  );

  const totalMatches = results.reduce((sum, result) => sum + result.totalMatches, 0);
  const totalHits = results.reduce((sum, result) => sum + result.hitCount, 0);

  return {
    results,
    totalMatches,
    totalHits,
    averageHitRate:
      totalMatches === 0 ? 0 : Math.round((totalHits / totalMatches) * 1000) / 10,
  };
}
export async function inspectBacktestRounds(roundNumbers: string[]) {
  const results = await Promise.all(
    roundNumbers.map(async (roundNumber) => {
      const result = await runBacktestForRound(roundNumber);

      return {
        roundNumber,
        round: result.round,
        totalMatches: result.totalMatches,
        hitCount: result.hitCount,
        hitRate: result.hitRate,
        isJleagueCandidate: result.totalMatches > 0,
      };
    })
  );

  return results;
}