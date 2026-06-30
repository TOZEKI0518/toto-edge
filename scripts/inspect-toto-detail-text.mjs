import fs from "fs";
import path from "path";
import * as cheerio from "cheerio";

const fileName = process.argv[2] ?? "totoRound-1637.html";
const filePath = path.join(process.cwd(), "data/snapshots", fileName);

if (!fs.existsSync(filePath)) {
  console.error(`File not found: ${filePath}`);
  process.exit(1);
}

const html = fs.readFileSync(filePath, "utf8");
const $ = cheerio.load(html);

const text = $.text().replace(/\s+/g, " ").trim();

console.log(`File: ${fileName}`);
console.log(`HTML length: ${html.length}`);
console.log(`Text length: ${text.length}`);

const keywords = [
  "開催回",
  "第1637回",
  "指定試合",
  "試合",
  "ホーム",
  "アウェイ",
  "投票率",
  "払戻",
  "当せん",
  "1等",
  "toto",
  "鹿",
  "浦",
  "神",
  "ＦＣ",
  "FC",
  "横浜",
  "川崎",
];

for (const keyword of keywords) {
  const index = text.indexOf(keyword);
  console.log(`\n--- ${keyword} ---`);

  if (index === -1) {
    console.log("not found");
    continue;
  }

  console.log(text.slice(Math.max(0, index - 150), Math.min(text.length, index + 400)));
}

console.log("\n=== Tables preview ===");
$("table").each((i, table) => {
  const tableText = $(table).text().replace(/\s+/g, " ").trim();
  console.log(`[table ${i}] length=${tableText.length}`);
  console.log(tableText.slice(0, 500));
  console.log("");
});