import type {
  ConfidenceRank,
  MatchPrediction,
  PredictionPick,
  PredictionReason,
  TotoMatch,
} from "@/types/toto";

const clamp = (value: number, min: number, max: number) => {
  return Math.min(Math.max(value, min), max);
};

const round = (value: number) => Math.round(value);

const getConfidence = (topProbability: number, edge: number): ConfidenceRank => {
  if (topProbability >= 68 && edge >= 12) return "S";
  if (topProbability >= 62 && edge >= 8) return "A";
  if (topProbability >= 56 && edge >= 5) return "B+";
  if (topProbability >= 50) return "B";
  return "C";
};

const getPick = (home: number, draw: number, away: number): PredictionPick => {
  const max = Math.max(home, draw, away);
  if (max === home) return "HOME";
  if (max === away) return "AWAY";
  return "DRAW";
};

export function calculatePrediction(match: TotoMatch): MatchPrediction {
  const reasons: PredictionReason[] = [];

  const rankImpact = clamp(
    (match.awayStats.rank - match.homeStats.rank) * 2,
    -16,
    16
  );

  reasons.push({
    label: "順位差",
    impact: rankImpact,
    description:
      rankImpact >= 0
        ? `${match.homeTeam.shortName}が順位面で優位です。`
        : `${match.awayTeam.shortName}が順位面で優位です。`,
  });

  const recentImpact = clamp(
    (match.homeStats.recentFormPoints - match.awayStats.recentFormPoints) * 1.5,
    -15,
    15
  );

  reasons.push({
    label: "直近成績",
    impact: round(recentImpact),
    description:
      recentImpact >= 0
        ? `${match.homeTeam.shortName}の直近成績が上回っています。`
        : `${match.awayTeam.shortName}の直近成績が上回っています。`,
  });

  const homeRateImpact = clamp(
    (match.homeStats.homeWinRate - match.awayStats.awayWinRate) * 25,
    -12,
    12
  );

  reasons.push({
    label: "ホーム/アウェイ成績",
    impact: round(homeRateImpact),
    description:
      homeRateImpact >= 0
        ? `${match.homeTeam.shortName}のホーム適性が優位です。`
        : `${match.awayTeam.shortName}のアウェイ成績が優位です。`,
  });

  const goalDiffImpact = clamp(
    (match.homeStats.goalDifference - match.awayStats.goalDifference) * 0.7,
    -10,
    10
  );

  reasons.push({
    label: "得失点差",
    impact: round(goalDiffImpact),
    description:
      goalDiffImpact >= 0
        ? `${match.homeTeam.shortName}の得失点差が上回っています。`
        : `${match.awayTeam.shortName}の得失点差が上回っています。`,
  });

  const weatherImpact =
    match.weather?.condition.includes("雨") ||
    match.weather?.condition.includes("雪")
      ? -4
      : 0;

  if (weatherImpact !== 0) {
    reasons.push({
      label: "天候",
      impact: weatherImpact,
      description: "悪天候により波乱・引き分けリスクを加味しています。",
    });
  }

  const homeScore =
    50 + rankImpact + recentImpact + homeRateImpact + goalDiffImpact;
  const awayScore =
    50 - rankImpact - recentImpact - homeRateImpact - goalDiffImpact;

  let homeProbability = clamp(homeScore, 18, 76);
  let awayProbability = clamp(awayScore, 12, 58);

  let drawProbability =
    100 - homeProbability - awayProbability + weatherImpact * -1;

  drawProbability = clamp(drawProbability, 18, 34);

  const total = homeProbability + drawProbability + awayProbability;

  homeProbability = round((homeProbability / total) * 100);
  drawProbability = round((drawProbability / total) * 100);
  awayProbability = 100 - homeProbability - drawProbability;

  const expectedValues = match.voteRates
    ? {
        home: round(homeProbability - match.voteRates.home),
        draw: round(drawProbability - match.voteRates.draw),
        away: round(awayProbability - match.voteRates.away),
      }
    : undefined;

  const pick = getPick(homeProbability, drawProbability, awayProbability);

  const topProbability =
    pick === "HOME"
      ? homeProbability
      : pick === "DRAW"
        ? drawProbability
        : awayProbability;

  const topEdge = expectedValues
    ? pick === "HOME"
      ? expectedValues.home
      : pick === "DRAW"
        ? expectedValues.draw
        : expectedValues.away
    : 0;

  return {
    matchId: match.id,
    pick,
    confidence: getConfidence(topProbability, topEdge),
    probabilities: {
      home: homeProbability,
      draw: drawProbability,
      away: awayProbability,
    },
    expectedValues,
    reasons,
  };
}

export function getPickLabel(pick: PredictionPick) {
  if (pick === "HOME") return "HOME WIN";
  if (pick === "AWAY") return "AWAY WIN";
  return "DRAW";
}