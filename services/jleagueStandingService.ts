import { fetchHtml } from "@/services/totoFetcher";
import { getTotoSourceUrl } from "@/services/totoSourceService";
import {
  htmlToPlainText,
  parseJleagueStandings,
} from "@/services/jleagueStandingParser";
import type { TeamStanding } from "@/types/standing";

export async function getJleagueStandings(): Promise<TeamStanding[]> {
  const html = await fetchHtml(getTotoSourceUrl("jleagueStandings"));
  const text = htmlToPlainText(html);

  return parseJleagueStandings(text);
}