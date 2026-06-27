import { calculatePrediction, getPickLabel } from "@/lib/predictionEngine";
import type { TotoMatch } from "@/types/toto";

export type DashboardMatch = {
  id: number;
  home: string;
  away: string;
  prediction: string;
  probability: number;
  edge: number;
  confidence: string;
};

export function buildDashboardMatch(match: TotoMatch): DashboardMatch {
  const prediction = calculatePrediction(match);
  const pickKey = prediction.pick.toLowerCase() as "home" | "draw" | "away";

  return {
    id: match.matchNo,
    home: match.homeTeam.shortName,
    away: match.awayTeam.shortName,
    prediction: getPickLabel(prediction.pick),
    probability: prediction.probabilities[pickKey],
    edge: prediction.expectedValues?.[pickKey] ?? 0,
    confidence: prediction.confidence,
  };
}

export function getBestPick(matches: DashboardMatch[]): DashboardMatch {
  return [...matches].sort((a, b) => b.edge - a.edge)[0];
}