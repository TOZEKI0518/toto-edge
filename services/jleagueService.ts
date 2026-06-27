import { sampleMatches } from "@/data/sampleMatches";
import { currentMatches } from "@/data/generated/currentMatches";
import type { TotoMatch } from "@/types/toto";

export type JLeagueDataSource = "sample" | "generated" | "api" | "scraping";

export async function fetchTotoMatches(
  source: JLeagueDataSource = "sample"
): Promise<TotoMatch[]> {
  if (source === "generated") return currentMatches;
  if (source === "sample") return sampleMatches;
  if (source === "api") throw new Error("API data source is not implemented yet.");
  if (source === "scraping") throw new Error("Scraping data source is not implemented yet.");

  return sampleMatches;
}

export function getDataSource(): JLeagueDataSource {
  const source = process.env.NEXT_PUBLIC_DATA_SOURCE;

  if (
    source === "sample" ||
    source === "generated" ||
    source === "api" ||
    source === "scraping"
  ) {
    return source;
  }

  return "sample";
}