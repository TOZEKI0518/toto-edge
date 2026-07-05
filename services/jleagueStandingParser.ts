import * as cheerio from "cheerio";
import type { AnyNode } from "domhandler";
import { normalizeTeamName } from "@/lib/teamNameNormalizer";
import type { TeamStanding } from "@/types/standing";

const toNumber = (value: string | undefined) => {
  if (!value) return 0;
  const n = Number(value.replace(/[^\d-]/g, ""));
  return Number.isFinite(n) ? n : 0;
};

const extractTeamName = ($: cheerio.CheerioAPI, cell: AnyNode) => {
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
          matches: toNumber($(cells[4]).text()),
          wins: toNumber($(cells[5]).text()),
          draws:
            toNumber($(cells[6]).text()) + toNumber($(cells[7]).text()),
          losses:
            toNumber($(cells[8]).text()) + toNumber($(cells[9]).text()),
          goalsFor: toNumber($(cells[9]).text()),
          goalsAgainst: toNumber($(cells[10]).text()),
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