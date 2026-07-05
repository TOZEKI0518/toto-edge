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
          label: "暫定予測",
          score: 0,
          description:
            "Jリーグ順位表と照合できないため、暫定的に引き分け寄りとして表示しています。",
        },
        {
          label: "データ状態",
          score: 0,
          description: "対象試合またはチーム名が順位データと一致していません。",
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

  const winRateScore = clamp(
    ((home.wins ?? 0) - (away.wins ?? 0)) * 1.2,
    -10,
    10
  );

  factors.push({
    label: "勝利数差",
    score: Math.round(winRateScore),
    description:
      winRateScore >= 0
        ? `${home.teamName}の勝利数を評価しています。`
        : `${away.teamName}の勝利数を評価しています。`,
  });

  const attackScore = clamp(
    ((home.goalsFor ?? 0) - (away.goalsFor ?? 0)) * 0.5,
    -8,
    8
  );

  factors.push({
    label: "攻撃力",
    score: Math.round(attackScore),
    description:
      attackScore >= 0
        ? `${home.teamName}の得点力を評価しています。`
        : `${away.teamName}の得点力を評価しています。`,
  });

  const defenseScore = clamp(
    ((away.goalsAgainst ?? 0) - (home.goalsAgainst ?? 0)) * 0.5,
    -8,
    8
  );

  factors.push({
    label: "守備力",
    score: Math.round(defenseScore),
    description:
      defenseScore >= 0
        ? `${home.teamName}の失点の少なさを評価しています。`
        : `${away.teamName}の失点の少なさを評価しています。`,
  });

  const homeAdvantageScore = 5;
  factors.push({
    label: "ホーム補正",
    score: homeAdvantageScore,
    description: `${home.teamName}のホーム開催を加点しています。`,
  });

  const totalScore = Math.round(
    rankScore +
  pointsScore +
  goalDiffScore +
  winRateScore +
  attackScore +
  defenseScore +
  homeAdvantageScore
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