import fs from "fs";
import path from "path";
import * as cheerio from "cheerio";

const url = process.argv[2];

if (!url) {
  console.error("Usage: node scripts/inspect-jleague-standings.mjs <url>");
  process.exit(1);
}

const res = await fetch(url);
const html = await res.text();
const $ = cheerio.load(html);

console.log("URL:", url);
console.log("HTML length:", html.length);
console.log("Title:", $("title").text().trim());

console.log("\n=== Tables ===");
$("table").each((index, table) => {
  const text = $(table).text().replace(/\s+/g, " ").trim();
  console.log(`[table ${index}] length=${text.length}`);
  console.log(text.slice(0, 500));
  console.log("");
});

console.log("\n=== Rows containing 札幌 / 磐田 / 仙台 ===");
$("tr").each((index, tr) => {
  const text = $(tr).text().replace(/\s+/g, " ").trim();

  if (text.includes("札幌") || text.includes("磐田") || text.includes("仙台")) {
    console.log(`[tr ${index}]`);
    console.log(text);
    console.log("");
  }
});