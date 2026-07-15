import type { BacktestRoundResult } from "@/types/backtest";
import type { MatchOutcome } from "@/types/prediction";

const OUTCOMES = ["HOME", "DRAW", "AWAY"] as const;
const PICK_LABELS: Record<MatchOutcome, string> = {
  HOME: "H",
  DRAW: "D",
  AWAY: "A",
};

type OutcomeProbability = Record<MatchOutcome, number>;

export type OptimizerBacktestTicket = {
  rank: number;
  picks: string;
  hitCount: number;
  missCount: number;
  isFirstPrizeEquivalent: boolean;
  isSecondPrizeEquivalent: boolean;
  isThirdPrizeEquivalent: boolean;
  modelProbability: number;
};

export type OptimizerBacktestResult = {
  ticketCount: number;
  investmentYen: number;
  matchCount: number;
  actualPicks: string;
  maxHitCount: number;
  maxHitRate: number;
  firstPrizeEquivalentCount: number;
  secondPrizeEquivalentCount: number;
  thirdPrizeEquivalentCount: number;
  winningTicketCount: number;
  tickets: OptimizerBacktestTicket[];
  monetaryRoiAvailable: false;
  monetaryRoiNote: string;
};

type Candidate = {
  outcomes: MatchOutcome[];
  logProbability: number;
};

function probabilityDistribution(totalScore: number): OutcomeProbability {
  const homeWeight = Math.exp(totalScore / 18);
  const awayWeight = Math.exp(-totalScore / 18);
  const drawWeight = Math.exp(0.75 - Math.abs(totalScore) / 20);
  const total = homeWeight + drawWeight + awayWeight;

  return {
    HOME: homeWeight / total,
    DRAW: drawWeight / total,
    AWAY: awayWeight / total,
  };
}

function candidateSimilarity(left: Candidate, right: Candidate) {
  if (left.outcomes.length === 0) return 0;

  const same = left.outcomes.reduce(
    (count, outcome, index) =>
      count + Number(outcome === right.outcomes[index]),
    0
  );

  return same / left.outcomes.length;
}

function buildCandidates(result: BacktestRoundResult, beamWidth = 2500) {
  const matches = [...result.matches].sort(
    (left, right) => left.matchNo - right.matchNo
  );

  let beam: Candidate[] = [{ outcomes: [], logProbability: 0 }];

  for (const match of matches) {
    const probabilities = probabilityDistribution(match.totalScore);
    const expanded: Candidate[] = [];

    for (const candidate of beam) {
      for (const outcome of OUTCOMES) {
        expanded.push({
          outcomes: [...candidate.outcomes, outcome],
          logProbability:
            candidate.logProbability +
            Math.log(Math.max(probabilities[outcome], 1e-12)),
        });
      }
    }

    expanded.sort(
      (left, right) => right.logProbability - left.logProbability
    );
    beam = expanded.slice(0, beamWidth);
  }

  return beam;
}

function selectDiverseCandidates(
  candidates: Candidate[],
  ticketCount: number
) {
  const selected: Candidate[] = [];
  const remaining = [...candidates];

  while (selected.length < ticketCount && remaining.length > 0) {
    let bestIndex = 0;
    let bestScore = Number.NEGATIVE_INFINITY;

    remaining.forEach((candidate, index) => {
      const similarity =
        selected.length === 0
          ? 0
          : Math.max(
              ...selected.map((chosen) =>
                candidateSimilarity(candidate, chosen)
              )
            );

      const adjustedScore =
        candidate.logProbability - similarity * 0.35;

      if (adjustedScore > bestScore) {
        bestScore = adjustedScore;
        bestIndex = index;
      }
    });

    selected.push(remaining.splice(bestIndex, 1)[0]);
  }

  return selected;
}

export function runOptimizerTicketBacktest(
  result: BacktestRoundResult,
  ticketCount = 50
): OptimizerBacktestResult {
  const matches = [...result.matches].sort(
    (left, right) => left.matchNo - right.matchNo
  );

  if (matches.length === 0) {
    return {
      ticketCount: 0,
      investmentYen: 0,
      matchCount: 0,
      actualPicks: "",
      maxHitCount: 0,
      maxHitRate: 0,
      firstPrizeEquivalentCount: 0,
      secondPrizeEquivalentCount: 0,
      thirdPrizeEquivalentCount: 0,
      winningTicketCount: 0,
      tickets: [],
      monetaryRoiAvailable: false,
      monetaryRoiNote:
        "対象試合がないため、10口バックテストを実行できません。",
    };
  }

  const candidates = buildCandidates(result);
  const selected = selectDiverseCandidates(candidates, ticketCount);
  const actualOutcomes = matches.map((match) => match.actualOutcome);
  const actualPicks = actualOutcomes
    .map((outcome) => PICK_LABELS[outcome])
    .join("");

  const tickets = selected.map((candidate, index) => {
    const hitCount = candidate.outcomes.reduce(
      (count, outcome, matchIndex) =>
        count + Number(outcome === actualOutcomes[matchIndex]),
      0
    );
    const missCount = matches.length - hitCount;
    const isFullToto = matches.length === 13;

    return {
      rank: index + 1,
      picks: candidate.outcomes
        .map((outcome) => PICK_LABELS[outcome])
        .join(""),
      hitCount,
      missCount,
      isFirstPrizeEquivalent: isFullToto && hitCount === 13,
      isSecondPrizeEquivalent: isFullToto && hitCount === 12,
      isThirdPrizeEquivalent: isFullToto && hitCount === 11,
      modelProbability: Math.exp(candidate.logProbability),
    };
  });

  const maxHitCount = Math.max(
    0,
    ...tickets.map((ticket) => ticket.hitCount)
  );
  const firstPrizeEquivalentCount = tickets.filter(
    (ticket) => ticket.isFirstPrizeEquivalent
  ).length;
  const secondPrizeEquivalentCount = tickets.filter(
    (ticket) => ticket.isSecondPrizeEquivalent
  ).length;
  const thirdPrizeEquivalentCount = tickets.filter(
    (ticket) => ticket.isThirdPrizeEquivalent
  ).length;

  return {
    ticketCount: tickets.length,
    investmentYen: tickets.length * 100,
    matchCount: matches.length,
    actualPicks,
    maxHitCount,
    maxHitRate:
      matches.length === 0
        ? 0
        : Math.round((maxHitCount / matches.length) * 1000) / 10,
    firstPrizeEquivalentCount,
    secondPrizeEquivalentCount,
    thirdPrizeEquivalentCount,
    winningTicketCount:
      firstPrizeEquivalentCount +
      secondPrizeEquivalentCount +
      thirdPrizeEquivalentCount,
    tickets,
    monetaryRoiAvailable: false,
    monetaryRoiNote:
      "払戻金データをまだ保存していないため、金額ROIは未算出です。現時点では13/13・12/13・11/13の当せん相当口数を確認できます。",
  };
}

export function summarizeOptimizerTicketBacktests(
  results: BacktestRoundResult[],
  ticketCount = 50
) {
  const rounds = results.map((result) => ({
    round: result.round,
    result: runOptimizerTicketBacktest(result, ticketCount),
  }));

  const totalInvestmentYen = rounds.reduce(
    (sum, item) => sum + item.result.investmentYen,
    0
  );
  const winningRounds = rounds.filter(
    (item) => item.result.winningTicketCount > 0
  ).length;
  const bestRound = [...rounds].sort(
    (left, right) =>
      right.result.maxHitCount - left.result.maxHitCount
  )[0];

  return {
    rounds,
    totalInvestmentYen,
    winningRounds,
    evaluatedRounds: rounds.length,
    winningRoundRate:
      rounds.length === 0
        ? 0
        : Math.round((winningRounds / rounds.length) * 1000) / 10,
    averageBestHits:
      rounds.length === 0
        ? 0
        : Math.round(
            (rounds.reduce(
              (sum, item) => sum + item.result.maxHitCount,
              0
            ) /
              rounds.length) *
              10
          ) / 10,
    bestRound: bestRound?.round ?? null,
    bestHitCount: bestRound?.result.maxHitCount ?? 0,
  };
}
