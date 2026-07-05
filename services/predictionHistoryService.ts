import { buildPredictionInputs } from "@/services/predictionInputService";
import { predictFixture } from "@/services/fixturePredictionService";
import { getJleagueStandings } from "@/services/jleagueStandingService";
import { fetchHtml } from "@/services/totoFetcher";
import { parseTotoFixturesFromHtml } from "@/services/totoFixtureParser";
import { mapTotoResultToOutcome } from "@/services/backtestOutcomeMapper";

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

const ALGORITHM_VERSION = "0.8.0";

function getRoundDetailUrl(roundNumber: string) {
  return `https://store.toto-dream.com/dcs/subos/screen/pi04/spin011/PGSPIN01101LnkHoldCntLotResultLsttoto.form?popupDispDiv=disp&holdCntId=${roundNumber}`;
}

async function supabaseInsert<T>(table: string, rows: T[]) {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error("Supabase environment variables are not set.");
  }

  const res = await fetch(`${SUPABASE_URL}/rest/v1/${table}`, {
    method: "POST",
    headers: {
      apikey: SUPABASE_SERVICE_ROLE_KEY,
      Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
      "Content-Type": "application/json",
      Prefer: "return=representation",
    },
    body: JSON.stringify(rows),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to insert ${table}: ${res.status} ${text}`);
  }

  return res.json();
}

export async function savePredictionHistoryForRound(roundNumber: string) {
  const [standings, html] = await Promise.all([
    getJleagueStandings(),
    fetchHtml(getRoundDetailUrl(roundNumber)),
  ]);

  const fixtures = parseTotoFixturesFromHtml(html);
  const inputs = buildPredictionInputs(fixtures, standings).filter(
    (input) => input.homeStanding && input.awayStanding
  );

  const predictionRows = inputs
    .map((input) => {
      const prediction = predictFixture(input);
      const fixture = fixtures.find((item) => item.matchNo === prediction.matchNo);
      const actualOutcome = mapTotoResultToOutcome(fixture?.totoResult ?? "");

      if (!actualOutcome) return null;

      return {
        round_no: Number(roundNumber),
        match_no: prediction.matchNo,
        home_team: prediction.homeTeam,
        away_team: prediction.awayTeam,
        predicted_outcome: prediction.outcome,
        actual_outcome: actualOutcome,
        probability: prediction.probability,
        confidence: prediction.confidence,
        total_score: prediction.totalScore,
        hit: prediction.outcome === actualOutcome,
        algorithm_version: ALGORITHM_VERSION,
      };
    })
    .filter((row): row is NonNullable<typeof row> => row !== null);

  const hitCount = predictionRows.filter((row) => row.hit).length;
  const totalMatches = predictionRows.length;
  const hitRate =
    totalMatches === 0 ? 0 : Math.round((hitCount / totalMatches) * 1000) / 10;

  const [run] = await supabaseInsert("prediction_runs", [
    {
      round_no: Number(roundNumber),
      data_snapshot_round: Number(roundNumber) - 1,
      algorithm_version: ALGORITHM_VERSION,
      hit_count: hitCount,
      total_matches: totalMatches,
      hit_rate: hitRate,
    },
  ]);

  const rowsWithRunId = predictionRows.map((row) => ({
    ...row,
    run_id: run.id,
  }));

  if (rowsWithRunId.length > 0) {
    await supabaseInsert("prediction_history", rowsWithRunId);
  }

  return {
    runId: run.id,
    roundNumber,
    algorithmVersion: ALGORITHM_VERSION,
    hitCount,
    totalMatches,
    hitRate,
  };
}