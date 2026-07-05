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

  $("table").each((_, table) => {
    $(table)
      .find("tbody tr")
      .each((_, row) => {
        const cells = $(row).find("th, td").toArray();

        if (cells.length < 10) return;

        const hasMoveIcon = $(cells[0]).find("i").length > 0;
        const offset = hasMoveIcon ? 1 : 0;

        const rank = toNumber($(cells[offset]).text());
        const teamName = extractTeamName($, cells[offset + 1]);

        if (!rank || !teamName) return;

        standings.push({
          rank,
          teamName,
          points: toNumber($(cells[offset + 2]).text()),
          matches: toNumber($(cells[offset + 3]).text()),
          wins: toNumber($(cells[offset + 4]).text()),
          draws: toNumber($(cells[offset + 5]).text()),
          losses: toNumber($(cells[offset + 6]).text()),
          goalsFor: toNumber($(cells[offset + 7]).text()),
          goalsAgainst: toNumber($(cells[offset + 8]).text()),
          goalDifference: toNumber($(cells[offset + 9]).text()),
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