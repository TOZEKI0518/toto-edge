import { fetchHtml } from "@/services/totoFetcher";
import { getTotoSourceUrl } from "@/services/totoSourceService";
import {
  htmlToPlainText,
  parseYahooTotoRounds,
} from "@/services/yahooTotoParser";
import type { TotoRoundSummary } from "@/types/totoRound";

export async function getYahooTotoRounds(): Promise<TotoRoundSummary[]> {
  const html = await fetchHtml(getTotoSourceUrl("yahooSchedule"));
  const text = htmlToPlainText(html);

  return parseYahooTotoRounds(text);
}

export async function getLatestYahooTotoRound(): Promise<TotoRoundSummary | null> {
  const rounds = await getYahooTotoRounds();

  const openRound = rounds.find((round) => round.status === "open");
  if (openRound) return openRound;

  return rounds[0] ?? null;
}