import { fetchHtml } from "@/services/totoFetcher";
import { parseJleagueStandingsFromHtml } from "@/services/jleagueStandingParser";
import type { TeamStanding } from "@/types/standing";

const standingUrls = [
  "https://www.jleague.jp/standings/j1/",
  "https://www.jleague.jp/standings/j2/",
  "https://www.jleague.jp/standings/j3/",
];

export async function getJleagueStandings(): Promise<TeamStanding[]> {
  const results = await Promise.all(
    standingUrls.map(async (url) => {
      const html = await fetchHtml(url);
      return parseJleagueStandingsFromHtml(html);
    })
  );

  return results.flat();
}