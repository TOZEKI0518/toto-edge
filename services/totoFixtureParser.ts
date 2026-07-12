import * as cheerio from "cheerio";
import type { TotoFixture } from "@/types/fixture";

const clean = (value: string) => value.replace(/\s+/g, " ").trim();
const cleanTeam = (value: string) => value.replace(/\s+/g, "").trim();

export function parseTotoFixturesFromHtml(html: string): TotoFixture[] {
  if (!html) return [];

  const $ = cheerio.load(html);
  const fixtures: TotoFixture[] = [];

  $("table").each((_, table) => {
    $(table)
      .find("tbody tr, tr")
      .each((__, row) => {
        const cells = $(row)
          .find("th, td")
          .map((___, cell) => clean($(cell).text()))
          .get()
          .filter(Boolean);

        if (cells.length < 6) return;

        const matchNoIndex = cells.findIndex((cell) => /^\d{1,2}$/.test(cell));
        if (matchNoIndex < 0) return;

        const matchNo = Number(cells[matchNoIndex]);
        const homeTeam = cleanTeam(cells[matchNoIndex + 1] ?? cells[3] ?? "");
        const awayTeam = cleanTeam(cells[matchNoIndex + 3] ?? cells[5] ?? "");
        const totoResult = cells.find((cell) => /^[012]$/.test(cell));

        if (!matchNo || !homeTeam || !awayTeam) return;

        if (fixtures.some((fixture) => fixture.matchNo === matchNo)) return;

        fixtures.push({
          matchNo,
          homeTeam,
          awayTeam,
          kickoffAt: cells[0],
          venue: cells[1],
          totoResult,
        });
      });
  });

  return fixtures.sort((a, b) => a.matchNo - b.matchNo);
}
