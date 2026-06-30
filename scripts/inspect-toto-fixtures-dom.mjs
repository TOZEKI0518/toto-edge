import fs from "fs";
import path from "path";
import * as cheerio from "cheerio";

const fileName = process.argv[2] ?? "yahooSchedule.html";

const filePath = path.join(process.cwd(), "data/snapshots", fileName);

if (!fs.existsSync(filePath)) {
  console.error(`Snapshot not found: ${filePath}`);
  console.error("Run: node scripts/save-source-snapshots.mjs");
  process.exit(1);
}

const html = fs.readFileSync(filePath, "utf8");
const $ = cheerio.load(html);

console.log(`File: ${fileName}`);
console.log(`HTML length: ${html.length}`);

console.log("\n=== Links containing toto / official / result ===");

$("a").each((i, a) => {
  const text = $(a).text().replace(/\s+/g, " ").trim();
  const href = $(a).attr("href") ?? "";

  if (
    text.includes("くじ結果") ||
    text.includes("第") ||
    href.includes("toto") ||
    href.includes("sportskuji") ||
    href.includes("official")
  ) {
    console.log(`[link ${i}]`);
    console.log(`text: ${text}`);
    console.log(`href: ${href}`);
    console.log("");
  }
});

console.log("\n=== Text around team keywords ===");

const plainText = $.text().replace(/\s+/g, " ").trim();
const keywords = ["鹿島", "浦和", "神戸", "FC東京", "横浜", "川崎"];

for (const keyword of keywords) {
  const index = plainText.indexOf(keyword);

  console.log(`--- ${keyword} ---`);

  if (index === -1) {
    console.log("not found");
    console.log("");
    continue;
  }

  console.log(
    plainText.slice(Math.max(0, index - 120), Math.min(plainText.length, index + 240))
  );
  console.log("");
}