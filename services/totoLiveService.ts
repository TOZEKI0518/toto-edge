import { fetchHtml } from "@/services/totoFetcher";
import { parseTotoPage } from "@/services/totoParser";
import {
  getTotoSourceUrl,
  type TotoSourceKey,
} from "@/services/totoSourceService";
import type { SourceFetchResult } from "@/types/source";

export async function getTotoLivePageSummary(
  sourceKey: TotoSourceKey = "rakutenSchedule"
): Promise<SourceFetchResult> {
  const url = getTotoSourceUrl(sourceKey);

  try {
    const html = await fetchHtml(url);
    const parsed = parseTotoPage(html);

    return {
      sourceKey,
      url,
      title: parsed.title,
      textLength: parsed.textLength,
      fetchedAt: new Date().toISOString(),
      ok: true,
    };
  } catch (error) {
    return {
      sourceKey,
      url,
      title: "Fetch failed",
      textLength: 0,
      fetchedAt: new Date().toISOString(),
      ok: false,
      errorMessage:
        error instanceof Error ? error.message : "Unknown fetch error",
    };
  }
}