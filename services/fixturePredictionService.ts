import type { PredictionInput } from "@/types/fixture";
import type {
  FixturePrediction,
  MatchOutcome,
  PredictionFactor,
} from "@/types/prediction";

const clamp = (value: number, min: number, max: number) => {
  return Math.min(Math.max(value, min), max);
};

const getOutcome = (score: number): MatchOutcome => {
  if (score >= 12) return "HOME";
  if (score <= -12) return "AWAY";
  return "DRAW";
};

const getProbability = (score: number) => {
  return Math.round(clamp(50 + Math.abs(score) * 1.2, 38, 78));
};

const getConfidence = (probability: number): FixturePrediction["confidence"] => {
  if (probability >= 70) return "S";
  if (probability >= 62) return "A";
  if (probability >= 55) return "B";
  return "C";
};

export function predictFixture(input: PredictionInput): FixturePrediction {
  const home = input.homeStanding;
  const away = input.awayStanding;

  const factors: PredictionFactor[] = [];

  if (!home || !away) {
    return {
      matchNo: input.fixture.matchNo,
      homeTeam: input.fixture.homeTeam,
      awayTeam: input.fixture.awayTeam,
      outcome: "DRAW",
      probability: 40,
      confidence: "C",
      totalScore: 0,
      factors: [
        {
          label: "データ不足",
          score: 0,
          description: "順位表データと照合できませんでした。",
        },
      ],
    };
  }

  const rankScore = clamp((away.rank - home.rank) * 2.5, -20, 20);
  factors.push({
    label: "順位差",
    score: Math.round(rankScore),
    description:
      rankScore >= 0
        ? `${home.teamName}が順位面で優位です。`
        : `${away.teamName}が順位面で優位です。`,
  });

  const pointsScore = clamp((home.points - away.points) * 0.8, -18, 18);
  factors.push({
    label: "勝点差",
    score: Math.round(pointsScore),
    description:
      pointsScore >= 0
        ? `${home.teamName}が勝点で上回っています。`
        : `${away.teamName}が勝点で上回っています。`,
  });

  const goalDiffScore = clamp(
    (home.goalDifference - away.goalDifference) * 0.7,
    -16,
    16
  );
  factors.push({
    label: "得失点差",
    score: Math.round(goalDiffScore),
    description:
      goalDiffScore >= 0
        ? `${home.teamName}の得失点差が上回っています。`
        : `${away.teamName}の得失点差が上回っています。`,
  });

  const homeAdvantageScore = 5;
  factors.push({
    label: "ホーム補正",
    score: homeAdvantageScore,
    description: `${home.teamName}のホーム開催を加点しています。`,
  });

  const totalScore = Math.round(
    rankScore + pointsScore + goalDiffScore + homeAdvantageScore
  );

  const outcome = getOutcome(totalScore);
  const probability = getProbability(totalScore);

  return {
    matchNo: input.fixture.matchNo,
    homeTeam: input.fixture.homeTeam,
    awayTeam: input.fixture.awayTeam,
    outcome,
    probability,
    confidence: getConfidence(probability),
    totalScore,
    factors,
  };
}

export function predictFixtures(
  inputs: PredictionInput[]
): FixturePrediction[] {
  return inputs.map(predictFixture);
}

export function getOutcomeLabel(outcome: MatchOutcome): string {
  if (outcome === "HOME") return "ホーム勝ち";
  if (outcome === "AWAY") return "アウェイ勝ち";
  return "引き分け";
}