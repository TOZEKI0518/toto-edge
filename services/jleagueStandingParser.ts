import type { TeamStanding } from "@/types/standing";

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

export function parseJleagueStandings(text: string): TeamStanding[] {
  const normalized = text.replace(/\s+/g, " ");
  const standings: TeamStanding[] = [];

  const teamNames = [
    "鹿島アントラーズ",
    "水戸ホーリーホック",
    "浦和レッズ",
    "ジェフユナイテッド千葉",
    "柏レイソル",
    "ＦＣ東京",
    "FC東京",
    "東京ヴェルディ",
    "ＦＣ町田ゼルビア",
    "FC町田ゼルビア",
    "川崎フロンターレ",
    "横浜Ｆ・マリノス",
    "横浜F・マリノス",
    "清水エスパルス",
    "名古屋グランパス",
    "京都サンガF.C.",
    "京都サンガＦ.Ｃ.",
    "ガンバ大阪",
    "セレッソ大阪",
    "ヴィッセル神戸",
    "ファジアーノ岡山",
    "サンフレッチェ広島",
    "アビスパ福岡",
    "Ｖ・ファーレン長崎",
    "V・ファーレン長崎",
  ];

  for (const teamName of teamNames) {
    const index = normalized.indexOf(teamName);
    if (index === -1) continue;

    const before = normalized.slice(Math.max(0, index - 80), index);
    const after = normalized.slice(index, index + 180);

    const rankMatch = before.match(/(\d{1,2})\s*$/);
    const numbers = after.match(/-?\d+/g)?.map(Number) ?? [];

    standings.push({
      rank: rankMatch ? Number(rankMatch[1]) : standings.length + 1,
      teamName,
      points: numbers[0] ?? 0,
      goalDifference: numbers.at(-1) ?? 0,
    });
  }

  return standings;
}