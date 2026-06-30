import * as cheerio from "cheerio";
import type { TotoFixture } from "@/types/fixture";

const normalizeHalfWidthTeamName = (name: string) =>
  name.replace(/\s+/g, "").trim();

export function parseTotoFixturesFromHtml(html: string): TotoFixture[] {
  const $ = cheerio.load(html);
  const fixtures: TotoFixture[] = [];

  const table = $("table").eq(2);

  table.find("tbody tr").each((_, row) => {
    const cells = $(row)
      .find("th, td")
      .map((_, cell) => $(cell).text().replace(/\s+/g, " ").trim())
      .get()
      .filter(Boolean);

    if (cells.length < 6) return;

    const matchNo = Number(cells[2]);
    const homeTeam = normalizeHalfWidthTeamName(cells[3]);
    const awayTeam = normalizeHalfWidthTeamName(cells[5]);

    if (!matchNo || !homeTeam || !awayTeam) return;

    fixtures.push({
      matchNo,
      homeTeam,
      awayTeam,
      kickoffAt: cells[0],
      venue: cells[1],
    });
  });

  return fixtures;
}