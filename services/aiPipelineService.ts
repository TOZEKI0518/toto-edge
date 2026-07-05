import { savePredictionHistoryForRound } from "@/services/predictionHistoryService";
import { saveCurrentStandingsSnapshot } from "@/services/standingsHistoryService";

export async function runAiPipelineForRound(roundNumber: string) {
  const roundNo = Number(roundNumber);

  if (!roundNo || !Number.isInteger(roundNo)) {
    throw new Error("roundNumber must be an integer.");
  }

  const snapshotRoundNo = roundNo - 1;

  const standingsSnapshot = await saveCurrentStandingsSnapshot(snapshotRoundNo);

  const predictionRun = await savePredictionHistoryForRound(roundNumber);

  return {
    roundNumber,
    snapshotRoundNo,
    standingsSnapshot,
    predictionRun,
  };
}