export const TOTO_SOURCE_URLS = {
  yahooSchedule: "https://toto.yahoo.co.jp/schedule/toto",
  rakutenSchedule: "https://toto.rakuten.co.jp/toto/schedule/",
  jleagueMatches: "https://www.jleague.jp/match/",
  jleagueStandings: "https://www.jleague.jp/standings/j1/",
} as const;

export type TotoSourceKey = keyof typeof TOTO_SOURCE_URLS;

export function getTotoSourceUrl(key: TotoSourceKey): string {
  return TOTO_SOURCE_URLS[key];
}