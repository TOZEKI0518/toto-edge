
import { supabaseSelect, supabaseUpsert } from "@/lib/supabase-rest";
import { fetchHtml } from "@/services/totoFetcher";
import { parseTotoFixturesFromHtml } from "@/services/totoFixtureParser";
import type { TotoFixture } from "@/types/fixture";
import { withLeague } from "@/services/leagueClassifier";

type TotoFixtureCacheRow = {
  round_no: number;
  match_no: number;
  home_team: string;
  away_team: string;
  kickoff_at: string | null;
  venue: string | null;
  toto_result: string | null;
};

export type RoundFixtures = {
  roundNo: number;
  fixtures: TotoFixture[];
  source: "memory" | "db" | "web" | "empty";
};

const memoryCache = new Map<number, RoundFixtures>();
const WEB_CONCURRENCY_DELAY_MS = 80;

function getRoundDetailUrl(roundNo: number) {
  return `https://store.toto-dream.com/dcs/subos/screen/pi04/spin011/PGSPIN01101LnkHoldCntLotResultLsttoto.form?popupDispDiv=disp&holdCntId=${roundNo}`;
}

function toFixture(row: TotoFixtureCacheRow): TotoFixture {
  return withLeague({
    matchNo: row.match_no,
    homeTeam: row.home_team,
    awayTeam: row.away_team,
    kickoffAt: row.kickoff_at ?? undefined,
    venue: row.venue ?? undefined,
    totoResult: row.toto_result ?? undefined,
  });
}

function toRow(roundNo: number, fixture: TotoFixture): TotoFixtureCacheRow {
  return {
    round_no: roundNo,
    match_no: fixture.matchNo,
    home_team: fixture.homeTeam,
    away_team: fixture.awayTeam,
    kickoff_at: fixture.kickoffAt ?? null,
    venue: fixture.venue ?? null,
    toto_result: fixture.totoResult ?? null,
  };
}

function chunk<T>(items: T[], size: number) {
  const chunks: T[][] = [];
  for (let i = 0; i < items.length; i += size) chunks.push(items.slice(i, i + size));
  return chunks;
}

async function loadCachedRows(roundNos: number[]) {
  const unique = Array.from(new Set(roundNos)).filter(Number.isInteger);
  if (unique.length === 0) return new Map<number, TotoFixture[]>();

  const rows: TotoFixtureCacheRow[] = [];
  for (const part of chunk(unique, 80)) {
    const query = `select=round_no,match_no,home_team,away_team,kickoff_at,venue,toto_result&round_no=in.(${part.join(",")})&order=round_no.asc,match_no.asc`;
    rows.push(...(await supabaseSelect<TotoFixtureCacheRow>("toto_round_fixtures", query)));
  }

  const grouped = new Map<number, TotoFixture[]>();
  for (const row of rows) {
    const list = grouped.get(row.round_no) ?? [];
    list.push(toFixture(row));
    grouped.set(row.round_no, list);
  }
  return grouped;
}

async function saveFixturesToCache(roundNo: number, fixtures: TotoFixture[]) {
  if (fixtures.length === 0) return;
  await supabaseUpsert(
    "toto_round_fixtures",
    fixtures.map((fixture) => toRow(roundNo, fixture)),
    "round_no,match_no"
  );
}

async function fetchFixturesFromWeb(roundNo: number): Promise<TotoFixture[]> {
  const html = await fetchHtml(getRoundDetailUrl(roundNo));
  return parseTotoFixturesFromHtml(html).map(withLeague);
}

export async function loadRoundFixtures(roundNo: number): Promise<RoundFixtures> {
  const cached = memoryCache.get(roundNo);
  if (cached) return cached;

  const dbRows = await loadCachedRows([roundNo]);
  const dbFixtures = dbRows.get(roundNo) ?? [];
  if (dbFixtures.length > 0) {
    const result = { roundNo, fixtures: dbFixtures, source: "db" as const };
    memoryCache.set(roundNo, result);
    return result;
  }

  const webFixtures = await fetchFixturesFromWeb(roundNo);
  const result = {
    roundNo,
    fixtures: webFixtures,
    source: webFixtures.length > 0 ? ("web" as const) : ("empty" as const),
  };
  memoryCache.set(roundNo, result);
  await saveFixturesToCache(roundNo, webFixtures);
  return result;
}

export async function loadManyRoundFixtures(
  roundNos: number[],
  options?: { maxMissingFetches?: number; concurrency?: number }
): Promise<RoundFixtures[]> {
  const unique = Array.from(new Set(roundNos)).filter(Number.isInteger);
  const results = new Map<number, RoundFixtures>();

  for (const roundNo of unique) {
    const cached = memoryCache.get(roundNo);
    if (cached) results.set(roundNo, cached);
  }

  const notInMemory = unique.filter((roundNo) => !results.has(roundNo));
  const dbRows = await loadCachedRows(notInMemory);

  for (const [roundNo, fixtures] of dbRows.entries()) {
    if (fixtures.length === 0) continue;
    const result = { roundNo, fixtures, source: "db" as const };
    memoryCache.set(roundNo, result);
    results.set(roundNo, result);
  }

  const missing = unique.filter((roundNo) => !results.has(roundNo));
  const maxMissingFetches = options?.maxMissingFetches ?? 30;
  const concurrency = Math.max(1, Math.min(options?.concurrency ?? 4, 8));
  const fetchTargets = missing.slice(0, maxMissingFetches);

  let cursor = 0;
  async function worker() {
    while (cursor < fetchTargets.length) {
      const index = cursor++;
      const roundNo = fetchTargets[index];
      if (index > 0) {
        await new Promise((resolve) => setTimeout(resolve, WEB_CONCURRENCY_DELAY_MS));
      }
      const loaded = await loadRoundFixtures(roundNo);
      results.set(roundNo, loaded);
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, fetchTargets.length) }, () => worker()));

  for (const roundNo of missing.slice(maxMissingFetches)) {
    const result = { roundNo, fixtures: [] as TotoFixture[], source: "empty" as const };
    results.set(roundNo, result);
  }

  return unique.map((roundNo) => results.get(roundNo) ?? { roundNo, fixtures: [], source: "empty" as const });
}
