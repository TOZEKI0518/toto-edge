import { getTotoSourceUrl } from "@/services/totoSourceService";
import { fetchHtml } from "@/services/totoFetcher";
import { parseTotoPage } from "@/services/totoParser";

export async function getTotoLivePageSummary() {
  const url = getTotoSourceUrl("rakutenSchedule");
  const html = await fetchHtml(url);
  return parseTotoPage(html);
}