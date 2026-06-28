import { fetchHtml } from "@/services/totoFetcher";
import { getTotoSourceUrl } from "@/services/totoSourceService";
import { parseJleagueStandingsFromHtml } from "@/services/jleagueStandingParser";
import type { TeamStanding } from "@/types/standing";

export async function getJleagueStandings(): Promise<TeamStanding[]> {
  const html = await fetchHtml(getTotoSourceUrl("jleagueStandings"));
  return parseJleagueStandingsFromHtml(html);
}