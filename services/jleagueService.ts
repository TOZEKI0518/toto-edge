import { sampleMatches } from "@/data/sampleMatches";
import type { TotoMatch } from "@/types/toto";

export type JLeagueDataSource = "sample" | "api" | "scraping";

export async function fetchTotoMatches(
  source: JLeagueDataSource = "sample"
): Promise<TotoMatch[]> {
  if (source === "sample") {
    return sampleMatches;
  }

  if (source === "api") {
    throw new Error("API data source is not implemented yet.");
  }

  if (source === "scraping") {
    throw new Error("Scraping data source is not implemented yet.");
  }

  return sampleMatches;
}