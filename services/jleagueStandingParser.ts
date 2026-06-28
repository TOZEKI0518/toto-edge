import * as cheerio from "cheerio";
import type { TeamStanding } from "@/types/standing";

const normalizeTeamName = (name: string) =>
  name
    .replace(/\s+/g, "")
    .replace(/Ｆ/g, "F")
    .replace(/Ｃ/g, "C")
    .replace(/Ｖ/g, "V")
    .replace(/．/g, ".")
    .trim();

const toNumber = (value: string | undefined) => {
  if (!value) return 0;
  const n = Number(value.replace(/[^\d-]/g, ""));
  return Number.isFinite(n) ? n : 0;
};

export function parseJleagueStandingsFromHtml(html: string): TeamStanding[] {
  const $ = cheerio.load(html);
  const standings: TeamStanding[] = [];

  $("table tbody tr").each((_, row) => {
    const cells = $(row)
      .find("th, td")
      .map((_, cell) => $(cell).text().trim())
      .get()
      .filter(Boolean);

    if (cells.length < 4) return;

    const rank = toNumber(cells[0]);
    const teamName = normalizeTeamName(cells[1]);

    if (!rank || !teamName) return;

    const numericCells = cells.map(toNumber);
    const points = numericCells.find((n, index) => index > 1 && n > 0) ?? 0;
    const goalDifference = numericCells.at(-1) ?? 0;

    standings.push({
      rank,
      teamName,
      points,
      goalDifference,
    });
  });

  const unique = new Map<string, TeamStanding>();

  for (const standing of standings) {
    if (!unique.has(standing.teamName)) {
      unique.set(standing.teamName, standing);
    }
  }

  return [...unique.values()].sort((a, b) => a.rank - b.rank);
}

export function htmlToPlainText(html: string): string {
  const $ = cheerio.load(html);
  return $.text().replace(/\s+/g, " ").trim();
}

export function parseJleagueStandings(text: string): TeamStanding[] {
  return [];
}