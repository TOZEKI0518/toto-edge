import fs from "fs";
import path from "path";

const filePath = path.join(
  process.cwd(),
  "data/snapshots/jleagueStandings.html"
);

if (!fs.existsSync(filePath)) {
  console.error("Snapshot not found. Run: node scripts/save-source-snapshots.mjs");
  process.exit(1);
}

const html = fs.readFileSync(filePath, "utf8");

const text = html
  .replace(/<script[\s\S]*?<\/script>/gi, "")
  .replace(/<style[\s\S]*?<\/style>/gi, " ")
  .replace(/<[^>]+>/g, " ")
  .replace(/&nbsp;/g, " ")
  .replace(/&amp;/g, "&")
  .replace(/\s+/g, " ")
  .trim();

const teamNames = [
  "鹿島アントラーズ",
  "浦和レッズ",
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

const rows = [];

for (const teamName of teamNames) {
  const index = text.indexOf(teamName);
  if (index === -1) continue;

  const before = text.slice(Math.max(0, index - 100), index);
  const after = text.slice(index, index + 220);
  const numbers = after.match(/-?\d+/g)?.map(Number) ?? [];

  rows.push({
    teamName,
    before,
    preview: after,
    numbers,
  });
}

console.table(
  rows.map((row) => ({
    teamName: row.teamName,
    numbers: row.numbers.join(", "),
    preview: row.preview.slice(0, 80),
  }))
);

console.log(`Found ${rows.length} team rows.`);