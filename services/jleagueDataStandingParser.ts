import * as cheerio from "cheerio";
import { normalizeTeamName } from "@/lib/teamNameNormalizer";
import type { TeamStanding } from "@/types/standing";

const toNumber = (value: string | undefined) => {
  if (!value) return 0;
  const n = Number(value.replace(/[^\d-]/g, ""));
  return Number.isFinite(n) ? n : 0;
};

export function parseJleagueDataStandingsFromHtml(html: string): TeamStanding[] {
  const $ = cheerio.load(html);
  const text = $.text().replace(/\s+/g, " ");

  const standings: TeamStanding[] = [];

  $("tr").each((_, row) => {
    const cells = $(row)
      .find("th, td")
      .map((_, cell) => $(cell).text().replace(/\s+/g, " ").trim())
      .get()
      .filter(Boolean);

    if (cells.length < 10) return;

    const rank = toNumber(cells[1]);
    const rawTeam = cells[2];
    const points = toNumber(cells[3]);

    if (!rank || !rawTeam || !points && cells[3] !== "0") return;

    standings.push({
      rank,
      teamName: normalizeTeamName(rawTeam),
      points,
      matches: toNumber(cells[4]),
      wins: toNumber(cells[5]),
      draws: toNumber(cells[6]),
      losses: toNumber(cells[7]),
      goalsFor: toNumber(cells[8]),
      goalsAgainst: toNumber(cells[9]),
      goalDifference: toNumber(cells[10]),
    });
  });

  if (standings.length > 0) return standings;

  // フォールバック：web取得テキストがtable化されない場合用
  const pattern =
    /グラフ\s+(\d+)\s+(.+?)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([+-]?\d+)/g;

  for (const match of text.matchAll(pattern)) {
    standings.push({
      rank: toNumber(match[1]),
      teamName: normalizeTeamName(match[2]),
      points: toNumber(match[3]),
      matches: toNumber(match[4]),
      wins: toNumber(match[5]),
      draws: toNumber(match[6]),
      losses: toNumber(match[7]),
      goalsFor: toNumber(match[8]),
      goalsAgainst: toNumber(match[9]),
      goalDifference: toNumber(match[10]),
    });
  }

  return standings;
}