import * as cheerio from "cheerio";
import { normalizeTeamName } from "@/lib/teamNameNormalizer";
import type { TeamStanding } from "@/types/standing";

const toNumber = (value: string | undefined) => {
  if (!value) return 0;
  const n = Number(value.replace(/[^\d-]/g, ""));
  return Number.isFinite(n) ? n : 0;
};

const extractTeamName = ($: cheerio.CheerioAPI, cell: cheerio.Element) => {
  const anchor = $(cell).find("a").first();
  const spanText = anchor.find("span").first().text().trim();

  if (spanText) return normalizeTeamName(spanText);

  return normalizeTeamName(anchor.text().trim() || $(cell).text().trim());
};

export function parseJleagueStandingsFromHtml(html: string): TeamStanding[] {
  const $ = cheerio.load(html);
  const standings: TeamStanding[] = [];

  $("table").each((tableIndex, table) => {
    if (tableIndex === 0) return;

    $(table)
      .find("tbody tr")
      .each((_, row) => {
        const cells = $(row).find("th, td").toArray();

        if (cells.length < 12) return;

        const rank = toNumber($(cells[1]).text());
        const teamName = extractTeamName($, cells[2]);
        const points = toNumber($(cells[3]).text());
        const goalDifference = toNumber($(cells[11]).text());

        if (!rank || !teamName) return;

        standings.push({
          rank,
          teamName,
          points,
          goalDifference,
        });
      });
  });

  return standings.sort((a, b) => a.rank - b.rank);
}

export function htmlToPlainText(html: string): string {
  const $ = cheerio.load(html);
  return $.text().replace(/\s+/g, " ").trim();
}

export function parseJleagueStandings(text: string): TeamStanding[] {
  return [];
}