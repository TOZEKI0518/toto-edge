import { buildPredictionInputs } from "@/services/predictionInputService";
import { predictFixture } from "@/services/fixturePredictionService";
import { getJleagueStandings } from "@/services/jleagueStandingService";
import { loadRoundFixtures } from "@/services/totoFixtureCacheService";
import { mapTotoResultToOutcome } from "@/services/backtestOutcomeMapper";
import { getTotoRoundInfo } from "@/services/totoRoundRepository";
import { buildDateCutStandings } from "@/services/dateCutStandingService";
import { isJLeagueFixture, withLeague } from "@/services/leagueClassifier";
import { supabaseSelect } from "@/lib/supabase-rest";
import type { BacktestRoundResult } from "@/types/backtest";
import type { TeamStanding } from "@/types/standing";

type DataSource = "date_cut" | "round_snapshot" | "snapshot" | "current" | "none";
type BacktestFilter = "jleague" | "all";

type StandingHistoryRow = {
  round_no: number | null;
  snapshot_date: string | null;
  team_name: string;
  rank: number | null;
  points: number | null;
  goal_difference: number | null;
  matches: number | null;
  wins: number | null;
  draws: number | null;
  losses: number | null;
  goals_for: number | null;
  goals_against: number | null;
};

type StandingLoadResult = {
  standings: TeamStanding[];
  snapshotDate: string | null;
  snapshotRoundNo: number | null;
  dataSource: DataSource;
  sourceRoundCount?: number;
};

function toStanding(row: StandingHistoryRow): TeamStanding {
  return {
    teamName: row.team_name,
    rank: row.rank ?? 0,
    points: row.points ?? 0,
    goalDifference: row.goal_difference ?? 0,
    matches: row.matches ?? undefined,
    wins: row.wins ?? undefined,
    draws: row.draws ?? undefined,
    losses: row.losses ?? undefined,
    goalsFor: row.goals_for ?? undefined,
    goalsAgainst: row.goals_against ?? undefined,
  };
}

function emptyStandingLoadResult(): StandingLoadResult {
  return {
    standings: [],
    snapshotDate: null,
    snapshotRoundNo: null,
    dataSource: "none",
    sourceRoundCount: 0,
  };
}

function buildRoundSnapshotResult(
  rows: StandingHistoryRow[],
  snapshotRoundNo: number
): StandingLoadResult {
  if (rows.length === 0) return emptyStandingLoadResult();

  return {
    standings: rows.map(toStanding),
    snapshotDate: rows.find((row) => row.snapshot_date)?.snapshot_date ?? null,
    snapshotRoundNo,
    dataSource: "round_snapshot",
  };
}

function buildDatedSnapshotResult(rows: StandingHistoryRow[]): StandingLoadResult {
  const validRows = rows.filter((row) => row.snapshot_date);

  if (validRows.length === 0) return emptyStandingLoadResult();

  const snapshotDate = validRows[0].snapshot_date;

  return {
    standings: validRows
      .filter((row) => row.snapshot_date === snapshotDate)
      .map(toStanding),
    snapshotDate,
    snapshotRoundNo: validRows[0].round_no ?? null,
    dataSource: "snapshot",
  };
}

async function getSnapshotByRoundNo(roundNumber: string): Promise<StandingLoadResult> {
  const roundNo = Number(roundNumber);

  if (!Number.isInteger(roundNo)) {
    return {
      ...emptyStandingLoadResult(),
    };
  }

  const rows = await supabaseSelect<StandingHistoryRow>(
    "standings_history",
    `select=*&round_no=eq.${roundNo}&order=team_name.asc`
  );

  return buildRoundSnapshotResult(rows, roundNo);
}

async function getSnapshotByMatchDate(matchDate: string | null): Promise<StandingLoadResult> {
  if (!matchDate) {
    return {
      ...emptyStandingLoadResult(),
    };
  }

  const rows = await supabaseSelect<StandingHistoryRow>(
    "standings_history",
    `select=*&snapshot_date=lte.${matchDate}&snapshot_date=not.is.null&order=snapshot_date.desc`
  );

  return buildDatedSnapshotResult(rows);
}

async function getLatestSnapshot(): Promise<StandingLoadResult> {
  const rows = await supabaseSelect<StandingHistoryRow>(
    "standings_history",
    "select=*&snapshot_date=not.is.null&order=snapshot_date.desc"
  );

  return buildDatedSnapshotResult(rows);
}

async function loadStandings(roundNumber: string): Promise<StandingLoadResult> {
  const roundNo = Number(roundNumber);

  // 新方式: 回号から開催日を引き、開催日より前に確定していたtoto結果だけで
  // チーム成績を再計算します。順位表スナップショット保存に依存しません。
  if (Number.isInteger(roundNo)) {
    const dateCut = await buildDateCutStandings(roundNo);

    if (dateCut.standings.length > 0) {
      return {
        standings: dateCut.standings,
        snapshotDate: dateCut.targetDate,
        snapshotRoundNo: null,
        dataSource: "date_cut",
        sourceRoundCount: dateCut.sourceRounds.length,
      };
    }
  }

  // 互換用: 既に回号別スナップショットが保存済みの環境では補助的に使います。
  const byExactRound = await getSnapshotByRoundNo(roundNumber);

  if (byExactRound.standings.length > 0) {
    return byExactRound;
  }

  // さらに古いDBで snapshot_date を持つ場合だけ、日付ベースを補助的に使います。
  const roundInfo = await getTotoRoundInfo(roundNumber);
  const byDate = await getSnapshotByMatchDate(roundInfo.matchDate);

  if (byDate.standings.length > 0) {
    return byDate;
  }

  const currentStandings = await getJleagueStandings();

  return {
    standings: currentStandings,
    snapshotDate: null,
    snapshotRoundNo: null,
    dataSource: currentStandings.length > 0 ? "current" : "none",
    sourceRoundCount: 0,
  };
}

function emptyResult(
  roundNumber: string,
  standingResult: StandingLoadResult
): BacktestRoundResult {
  return {
    round: `第${roundNumber}回`,
    totalFixtures: 0,
    totalMatches: 0,
    hitCount: 0,
    hitRate: 0,
    coverageRate: 0,
    matches: [],
    unmatchedTeams: [],
    snapshotDate: standingResult.snapshotDate,
    dataSource: standingResult.dataSource,
    snapshotRoundNo: standingResult.snapshotRoundNo,
    sourceRoundCount: standingResult.sourceRoundCount ?? 0,
  };
}

export async function runBacktestForRound(
  roundNumber: string,
  options?: { filter?: BacktestFilter }
): Promise<BacktestRoundResult> {
  const standingResult = await loadStandings(roundNumber);

  const loadedFixtures = await loadRoundFixtures(Number(roundNumber));
  const allFixtures = loadedFixtures.fixtures.map(withLeague);
  const fixtures = (options?.filter ?? "jleague") === "jleague"
    ? allFixtures.filter(isJLeagueFixture)
    : allFixtures;

  if (fixtures.length === 0) {
    return emptyResult(roundNumber, standingResult);
  }

  const inputs = buildPredictionInputs(fixtures, standingResult.standings);

  const validInputs = inputs.filter(
    (input) => input.homeStanding && input.awayStanding
  );

  const unmatchedTeams = Array.from(
    new Set(
      inputs.flatMap((input) => {
        const teams: string[] = [];

        if (!input.homeStanding) teams.push(input.fixture.homeTeam);
        if (!input.awayStanding) teams.push(input.fixture.awayTeam);

        return teams;
      })
    )
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
        league: fixture?.league ?? "Unknown",
        predictedOutcome: prediction.outcome,
        actualOutcome,
        probability: prediction.probability,
        confidence: prediction.confidence,
        totalScore: prediction.totalScore,
        isHit: prediction.outcome === actualOutcome,
        factors: prediction.factors,
      };
    })
    .filter((match): match is NonNullable<typeof match> => match !== null);

  const hitCount = matches.filter((match) => match.isHit).length;
  const totalMatches = matches.length;
  const totalFixtures = fixtures.length;

  return {
    round: `第${roundNumber}回`,
    totalFixtures,
    totalMatches,
    hitCount,
    hitRate:
      totalMatches === 0
        ? 0
        : Math.round((hitCount / totalMatches) * 1000) / 10,
    coverageRate:
      totalFixtures === 0
        ? 0
        : Math.round((totalMatches / totalFixtures) * 1000) / 10,
    matches,
    unmatchedTeams,
    snapshotDate: standingResult.snapshotDate,
    dataSource: standingResult.dataSource,
    snapshotRoundNo: standingResult.snapshotRoundNo,
    sourceRoundCount: standingResult.sourceRoundCount ?? 0,
  };
}

async function mapBacktestsWithConcurrency(roundNumbers: string[], concurrency = 4) {
  const results: PromiseSettledResult<BacktestRoundResult>[] = [];
  let cursor = 0;

  async function worker() {
    while (cursor < roundNumbers.length) {
      const index = cursor++;
      try {
        results[index] = {
          status: "fulfilled",
          value: await runBacktestForRound(roundNumbers[index]),
        };
      } catch (reason) {
        results[index] = { status: "rejected", reason };
      }
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(concurrency, Math.max(roundNumbers.length, 1)) }, () => worker())
  );

  return results;
}


async function mapBacktestsWithConcurrencyWithFilter(roundNumbers: string[], filter: BacktestFilter, concurrency = 4) {
  const results: PromiseSettledResult<BacktestRoundResult>[] = [];
  let cursor = 0;

  async function worker() {
    while (cursor < roundNumbers.length) {
      const index = cursor++;
      try {
        results[index] = {
          status: "fulfilled",
          value: await runBacktestForRound(roundNumbers[index], { filter }),
        };
      } catch (reason) {
        results[index] = { status: "rejected", reason };
      }
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(concurrency, Math.max(roundNumbers.length, 1)) }, () => worker())
  );

  return results;
}

function roundRate(hits: number, total: number) {
  return total === 0 ? 0 : Math.round((hits / total) * 1000) / 10;
}


function buildLeagueStats(results: BacktestRoundResult[]) {
  const map = new Map<string, { league: string; hits: number; total: number }>();

  for (const result of results) {
    for (const match of result.matches) {
      const league = match.league ?? "Unknown";
      const item = map.get(league) ?? { league, hits: 0, total: 0 };
      item.total += 1;
      if (match.isHit) item.hits += 1;
      map.set(league, item);
    }
  }

  return Array.from(map.values())
    .map((item) => ({ ...item, rate: roundRate(item.hits, item.total) }))
    .sort((a, b) => b.total - a.total);
}

export async function runBacktestSummary(roundNumbers: string[], options?: { filter?: BacktestFilter }) {
  const settled = await mapBacktestsWithConcurrencyWithFilter(roundNumbers, options?.filter ?? "jleague", 4);

  const results = settled
    .filter(
      (item): item is PromiseFulfilledResult<BacktestRoundResult> =>
        item.status === "fulfilled"
    )
    .map((item) => item.value)
    .filter((result) => result.totalMatches > 0);

  const totalMatches = results.reduce(
    (sum, result) => sum + result.totalMatches,
    0
  );

  const totalHits = results.reduce((sum, result) => sum + result.hitCount, 0);

  const teamMap = new Map<string, { team: string; hits: number; total: number }>();
  const outcomeMap = new Map<string, { outcome: string; hits: number; total: number }>();

  for (const result of results) {
    for (const match of result.matches) {
      for (const team of [match.homeTeam, match.awayTeam]) {
        const item = teamMap.get(team) ?? { team, hits: 0, total: 0 };
        item.total += 1;
        if (match.isHit) item.hits += 1;
        teamMap.set(team, item);
      }

      const outcome = match.predictedOutcome;
      const outcomeItem = outcomeMap.get(outcome) ?? { outcome, hits: 0, total: 0 };
      outcomeItem.total += 1;
      if (match.isHit) outcomeItem.hits += 1;
      outcomeMap.set(outcome, outcomeItem);
    }
  }

  const teamAccuracy = Array.from(teamMap.values())
    .filter((item) => item.total >= 3)
    .map((item) => ({ ...item, rate: roundRate(item.hits, item.total) }))
    .sort((a, b) => b.rate - a.rate || b.total - a.total);

  const outcomeAccuracy = Array.from(outcomeMap.values())
    .map((item) => ({ ...item, rate: roundRate(item.hits, item.total) }))
    .sort((a, b) => b.total - a.total);

  return {
    results,
    totalMatches,
    totalHits,
    averageHitRate: roundRate(totalHits, totalMatches),
    leagueStats: buildLeagueStats(results),
    topTeams: teamAccuracy.slice(0, 5),
    difficultTeams: teamAccuracy.slice(-5).reverse(),
    outcomeAccuracy,
  };
}

export async function inspectBacktestRounds(roundNumbers: string[]) {
  const settled = await mapBacktestsWithConcurrencyWithFilter(roundNumbers, "jleague", 4);

  return settled.map((item, index) => {
    const roundNumber = roundNumbers[index];

    if (item.status === "rejected") {
      return {
        roundNumber,
        round: `第${roundNumber}回`,
        totalFixtures: 0,
        totalMatches: 0,
        hitCount: 0,
        hitRate: 0,
        coverageRate: 0,
        isJleagueCandidate: false,
        snapshotDate: null,
        dataSource: "none" as DataSource,
        snapshotRoundNo: null,
        sourceRoundCount: 0,
      };
    }

    const result = item.value;

    return {
      roundNumber,
      round: result.round,
      totalFixtures: result.totalFixtures,
      totalMatches: result.totalMatches,
      hitCount: result.hitCount,
      hitRate: result.hitRate,
      coverageRate: result.coverageRate,
      isJleagueCandidate: result.totalMatches > 0,
      snapshotDate: result.snapshotDate ?? null,
      dataSource: result.dataSource ?? "none",
      snapshotRoundNo: result.snapshotRoundNo ?? null,
      sourceRoundCount: result.sourceRoundCount ?? 0,
    };
  });
}
