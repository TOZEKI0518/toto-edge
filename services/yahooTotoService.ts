import { fetchHtml } from "@/services/totoFetcher";
import { getTotoSourceUrl } from "@/services/totoSourceService";
import {
  htmlToPlainText,
  parseYahooTotoRounds,
} from "@/services/yahooTotoParser";

export async function getYahooTotoRounds() {
  const html = await fetchHtml(getTotoSourceUrl("yahooSchedule"));
  const text = htmlToPlainText(html);

  return parseYahooTotoRounds(text);
}