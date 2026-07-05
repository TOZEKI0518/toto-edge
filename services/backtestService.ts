import { buildPredictionInputs } from "@/services/predictionInputService";
import { predictFixtures } from "@/services/fixturePredictionService";
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
  const predictions = predictFixtures(inputs);

  const matches = predictions
    .map((prediction) => {
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