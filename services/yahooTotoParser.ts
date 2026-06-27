import type { TotoRoundSummary } from "@/types/totoRound";

export function parseYahooTotoRounds(text: string): TotoRoundSummary[] {
  const normalized = text.replace(/\s+/g, " ");

  const regex =
    /(販売中|終了)?\s*第(\d+)回\s*（(\d{4}\/\d{1,2}\/\d{1,2})-(\d{4}\/\d{1,2}\/\d{1,2})）/g;

  const rounds: TotoRoundSummary[] = [];

  for (const match of normalized.matchAll(regex)) {
    const rawStatus = match[1] ?? "unknown";

    rounds.push({
      round: `第${match[2]}回`,
      salesStart: match[3],
      salesEnd: match[4],
      status:
        rawStatus === "販売中"
          ? "open"
          : rawStatus === "終了"
            ? "closed"
            : "unknown",
    });
  }

  return rounds;
}

export function htmlToPlainText(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
}