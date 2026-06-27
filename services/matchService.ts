import {
  fetchTotoMatches,
  getDataSource,
} from "@/services/jleagueService";
import { getLatestYahooTotoRound } from "@/services/yahooTotoService";
import type { TotoMatch } from "@/types/toto";

export async function getMatches(): Promise<TotoMatch[]> {
  const source = getDataSource();
  return fetchTotoMatches(source);
}

export async function getCurrentTotoRound(): Promise<string> {
  try {
    const latestRound = await getLatestYahooTotoRound();
    return latestRound?.round ?? "第XXXX回";
  } catch {
    const matches = await getMatches();
    return matches[0]?.totoRound ?? "第XXXX回";
  }
}