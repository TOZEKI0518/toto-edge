import { sampleMatches } from "@/data/sampleMatches";
import type { TotoMatch } from "@/types/toto";

export async function getMatches(): Promise<TotoMatch[]> {
  return sampleMatches;
}

export async function getCurrentTotoRound(): Promise<string> {
  const matches = await getMatches();
  return matches[0]?.totoRound ?? "第XXXX回";
}