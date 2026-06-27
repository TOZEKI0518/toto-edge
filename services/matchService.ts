import {
  fetchTotoMatches,
  getDataSource,
} from "@/services/jleagueService";
import type { TotoMatch } from "@/types/toto";

export async function getMatches(): Promise<TotoMatch[]> {
  const source = getDataSource();
  return fetchTotoMatches(source);
}

export async function getCurrentTotoRound(): Promise<string> {
  const matches = await getMatches();
  return matches[0]?.totoRound ?? "第XXXX回";
}