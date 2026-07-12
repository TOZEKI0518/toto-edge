import { normalizeTeamName } from "@/lib/teamNameNormalizer";
import type { PredictionInput } from "@/types/fixture";
import type {
  FixturePrediction,
  MatchOutcome,
  PredictionFactor,
} from "@/types/prediction";

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

const round1 = (value: number) => Math.round(value * 10) / 10;

function weighted(score: number, weight: number) {
  return score * weight;
}

function formatSigned(value: number) {
  const rounded = round1(value);
  return rounded > 0 ? `+${rounded}` : String(rounded);
}

// v2.0 Stability Engine
// これまでのExcel比較では、特徴量を増やすほど精度が落ちる傾向がありました。
// そのため、予測の中核を「Home/Away ELO + 勝点/順位」に戻し、
// Head-to-Headは診断表示のみ、フォームは十分な試合数がある時だけ弱く効かせます。
const WEIGHTS = {
  sideElo: 1.1,
  totalElo: 0.55,
  points: 0.7,
  rank: 0.55,
  goalDiff: 0.45,
  winRate: 0.45,
  recentForm: 0.28,
  sideForm: 0.28,
  attackDefense: 0.25,
  homeAdvantage: 0.75,
};

function getResultRate(wins = 0, draws = 0, matches = 0) {
  if (!matches) return 0.5;
  return (wins + draws * 0.35) / matches;
}

function getDrawRate(draws = 0, matches = 0) {
  if (!matches) return 0;
  return draws / matches;
}

function getOutcome(params: {
  score: number;
  drawSignal: number;
  strongSideSignal: number;
}): MatchOutcome {
  const absScore = Math.abs(params.score);

  // v2.1 Uncertainty + Draw Calibration
  // 1630回の詳細を見ると、総合スコア差が小さいのにS/A判定で
  // Home/Awayへ振り切って外すケースが多かったため、
  // 「僅差ゾーン」は勝敗より引き分けを優先します。
  if (absScore <= 4) return "DRAW";
  if (absScore <= 7 && params.drawSignal >= 2.4 && params.strongSideSignal < 3.2) return "DRAW";
  if (absScore <= 10 && params.drawSignal >= 4.4 && params.strongSideSignal < 2.2) return "DRAW";

  return params.score >= 0 ? "HOME" : "AWAY";
}

function getProbability(score: number, drawSignal: number, outcome: MatchOutcome) {
  const certainty = Math.abs(score);

  if (outcome === "DRAW") {
    // Drawは元々難しいため、過信しない。
    return Math.round(clamp(43 + drawSignal * 2.4 - certainty * 0.35, 42, 57));
  }

  // 僅差なのに70%以上/S判定になる問題を修正。
  // 確信度は「スコア差」が十分に開いた時だけ上げます。
  return Math.round(clamp(48 + certainty * 0.95 - drawSignal * 0.75, 45, 74));
}

const getConfidence = (probability: number): FixturePrediction["confidence"] => {
  if (probability >= 72) return "S";
  if (probability >= 65) return "A";
  if (probability >= 56) return "B";
  return "C";
};

function getHeadToHeadDiagnostic(input: PredictionInput): PredictionFactor {
  const home = input.homeStanding;
  const away = input.awayStanding;

  if (!home || !away) {
    return {
      label: "対戦相性",
      score: 0,
      description: "対戦相性データが不足しています。",
    };
  }

  const awayKey = normalizeTeamName(away.teamName);
  const record = home.headToHead?.[awayKey];

  if (!record || record.matches === 0) {
    return {
      label: "対戦相性",
      score: 0,
      description: `${home.teamName}と${away.teamName}の過去対戦データは限定的です。今回はスコアに反映していません。`,
    };
  }

  const rate = getResultRate(record.wins, record.draws, record.matches);
  const diagnosticScore = clamp((rate - 0.5) * 10 + record.weightedScore * 3, -7, 7);

  return {
    label: "対戦相性",
    score: Math.round(diagnosticScore),
    description:
      `${home.teamName}視点では${record.wins}勝${record.draws}分${record.losses}敗です。` +
      " Excel比較で悪化要因になったため、今版では診断表示のみです。",
  };
}

function getDrawSignal(params: {
  eloDiff: number;
  sideEloDiff: number;
  pointsDiff: number;
  rankDiff: number;
  formDiff: number;
  sideFormDiff: number;
  homeDrawRate: number;
  awayDrawRate: number;
}) {
  let signal = 0;

  if (Math.abs(params.sideEloDiff) < 28) signal += 1.8;
  if (Math.abs(params.eloDiff) < 25) signal += 1.4;
  if (Math.abs(params.pointsDiff) <= 2) signal += 1.2;
  if (Math.abs(params.rankDiff) <= 2) signal += 0.9;
  if (Math.abs(params.formDiff) <= 2) signal += 0.7;
  if (Math.abs(params.sideFormDiff) <= 2) signal += 0.7;

  const averageDrawRate = (params.homeDrawRate + params.awayDrawRate) / 2;
  if (averageDrawRate >= 0.34) signal += 1.1;
  else if (averageDrawRate >= 0.27) signal += 0.6;

  return clamp(signal, 0, 6.5);
}

function getStrongSideSignal(params: {
  sideEloDiff: number;
  pointsDiff: number;
  rankDiff: number;
  winRateDiff: number;
}) {
  let signal = 0;
  if (Math.abs(params.sideEloDiff) >= 65) signal += 2.2;
  if (Math.abs(params.pointsDiff) >= 6) signal += 1.4;
  if (Math.abs(params.rankDiff) >= 5) signal += 1.1;
  if (Math.abs(params.winRateDiff) >= 0.18) signal += 1.0;
  return signal;
}

export function predictFixture(input: PredictionInput): FixturePrediction {
  const home = input.homeStanding;
  const away = input.awayStanding;

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
      ],
    };
  }

  const factors: PredictionFactor[] = [];
  const scoreParts: number[] = [];

  const homeMatches = home.matches ?? 0;
  const awayMatches = away.matches ?? 0;
  const enoughHomeSample = homeMatches >= 4;
  const enoughAwaySample = awayMatches >= 4;

  const homeElo = home.eloRating ?? 1500;
  const awayElo = away.eloRating ?? 1500;
  const eloDiff = homeElo - awayElo;
  const totalEloScore = weighted(clamp(eloDiff * 0.075, -14, 14), WEIGHTS.totalElo);
  scoreParts.push(totalEloScore);
  factors.push({
    label: "総合ELO",
    score: Math.round(totalEloScore),
    description:
      totalEloScore >= 0
        ? `${home.teamName}の総合ELOが高めです。調整後 ${formatSigned(totalEloScore)}。`
        : `${away.teamName}の総合ELOが高めです。調整後 ${formatSigned(totalEloScore)}。`,
  });

  const homeSideElo = home.homeEloRating ?? homeElo;
  const awaySideElo = away.awayEloRating ?? awayElo;
  const sideEloDiff = homeSideElo - awaySideElo;
  const sideEloScore = weighted(clamp(sideEloDiff * 0.095, -18, 18), WEIGHTS.sideElo);
  scoreParts.push(sideEloScore);
  factors.push({
    label: "Home/Away ELO",
    score: Math.round(sideEloScore),
    description:
      sideEloScore >= 0
        ? `${home.teamName}のホームELO（${Math.round(homeSideElo)}）を重視しています。`
        : `${away.teamName}のアウェイELO（${Math.round(awaySideElo)}）を重視しています。`,
  });

  const pointsDiff = home.points - away.points;
  const pointsScore = weighted(clamp(pointsDiff * 0.5, -12, 12), WEIGHTS.points);
  scoreParts.push(pointsScore);
  factors.push({
    label: "勝点差",
    score: Math.round(pointsScore),
    description:
      pointsScore >= 0
        ? `${home.teamName}が勝点で上回っています。`
        : `${away.teamName}が勝点で上回っています。`,
  });

  const rankDiff = away.rank - home.rank;
  const rankScore = weighted(clamp(rankDiff * 1.25, -12, 12), WEIGHTS.rank);
  scoreParts.push(rankScore);
  factors.push({
    label: "順位差",
    score: Math.round(rankScore),
    description:
      rankScore >= 0
        ? `${home.teamName}が順位面で優位です。`
        : `${away.teamName}が順位面で優位です。`,
  });

  const goalDiffDiff = home.goalDifference - away.goalDifference;
  const goalDiffScore = weighted(clamp(goalDiffDiff * 0.55, -10, 10), WEIGHTS.goalDiff);
  scoreParts.push(goalDiffScore);
  factors.push({
    label: "得失点差",
    score: Math.round(goalDiffScore),
    description:
      goalDiffScore >= 0
        ? `${home.teamName}の得失点差を評価しています。`
        : `${away.teamName}の得失点差を評価しています。`,
  });

  const homeWinRate = homeMatches > 0 ? (home.wins ?? 0) / homeMatches : 0.33;
  const awayWinRate = awayMatches > 0 ? (away.wins ?? 0) / awayMatches : 0.33;
  const winRateDiff = homeWinRate - awayWinRate;
  const winRateScore = weighted(clamp(winRateDiff * 28, -8, 8), WEIGHTS.winRate);
  scoreParts.push(winRateScore);
  factors.push({
    label: "勝率差",
    score: Math.round(winRateScore),
    description:
      winRateScore >= 0
        ? `${home.teamName}の勝率を評価しています。`
        : `${away.teamName}の勝率を評価しています。`,
  });

  const homeFormPoints = enoughHomeSample ? home.recentFormPoints ?? 0 : 0;
  const awayFormPoints = enoughAwaySample ? away.recentFormPoints ?? 0 : 0;
  const formDiff = homeFormPoints - awayFormPoints;
  const formScore = weighted(clamp(formDiff * 1.2, -10, 10), WEIGHTS.recentForm);
  scoreParts.push(formScore);
  factors.push({
    label: "直近5試合フォーム",
    score: Math.round(formScore),
    description:
      formScore >= 0
        ? `${home.teamName}の直近成績（${home.recentFormLabel || "-"}）を控えめに評価しています。`
        : `${away.teamName}の直近成績（${away.recentFormLabel || "-"}）を控えめに評価しています。`,
  });

  const homeSideForm = enoughHomeSample ? home.homeRecentFormPoints ?? 0 : 0;
  const awaySideForm = enoughAwaySample ? away.awayRecentFormPoints ?? 0 : 0;
  const sideFormDiff = homeSideForm - awaySideForm;
  const sideFormScore = weighted(clamp(sideFormDiff * 1.15, -9, 9), WEIGHTS.sideForm);
  scoreParts.push(sideFormScore);
  factors.push({
    label: "Home/Awayフォーム",
    score: Math.round(sideFormScore),
    description:
      sideFormScore >= 0
        ? `${home.teamName}のホーム直近成績（${home.homeRecentFormLabel || "-"}）を控えめに評価しています。`
        : `${away.teamName}のアウェイ直近成績（${away.awayRecentFormLabel || "-"}）を控えめに評価しています。`,
  });

  const attackDefenseRaw =
    ((home.goalsFor ?? 0) - (away.goalsFor ?? 0)) * 0.25 +
    ((away.goalsAgainst ?? 0) - (home.goalsAgainst ?? 0)) * 0.25;
  const attackDefenseScore = weighted(clamp(attackDefenseRaw, -6, 6), WEIGHTS.attackDefense);
  scoreParts.push(attackDefenseScore);
  factors.push({
    label: "攻守バランス",
    score: Math.round(attackDefenseScore),
    description:
      attackDefenseScore >= 0
        ? `${home.teamName}の攻守バランスをわずかに評価しています。`
        : `${away.teamName}の攻守バランスをわずかに評価しています。`,
  });

  const homeAdvantageScore = weighted(5, WEIGHTS.homeAdvantage);
  scoreParts.push(homeAdvantageScore);
  factors.push({
    label: "ホーム補正",
    score: Math.round(homeAdvantageScore),
    description: `${home.teamName}のホーム開催を加点しています。`,
  });

  const h2hFactor = getHeadToHeadDiagnostic(input);
  factors.push(h2hFactor);

  const homeDrawRate = getDrawRate(home.draws, homeMatches);
  const awayDrawRate = getDrawRate(away.draws, awayMatches);
  const drawSignal = getDrawSignal({
    eloDiff,
    sideEloDiff,
    pointsDiff,
    rankDiff,
    formDiff,
    sideFormDiff,
    homeDrawRate,
    awayDrawRate,
  });
  const strongSideSignal = getStrongSideSignal({
    sideEloDiff,
    pointsDiff,
    rankDiff,
    winRateDiff,
  });

  factors.push({
    label: "Draw Engine",
    score: Math.round(drawSignal),
    description:
      drawSignal >= 2.4
        ? "両チームの力差が小さいため、僅差ゾーンでは引き分け候補を広めに評価します。"
        : "片側優位が強い場合は勝敗を優先し、僅差の場合のみ引き分けを検討します。",
  });

  const totalScore = Math.round(scoreParts.reduce((sum, part) => sum + part, 0));
  const outcome = getOutcome({ score: totalScore, drawSignal, strongSideSignal });
  const probability = getProbability(totalScore, drawSignal, outcome);

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

export function predictFixtures(inputs: PredictionInput[]): FixturePrediction[] {
  return inputs.map(predictFixture);
}

export function getOutcomeLabel(outcome: MatchOutcome): string {
  if (outcome === "HOME") return "ホーム勝ち";
  if (outcome === "AWAY") return "アウェイ勝ち";
  return "引き分け";
}
