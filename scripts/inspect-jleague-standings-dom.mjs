import fs from "fs";
import path from "path";
import * as cheerio from "cheerio";

const filePath = path.join(
  process.cwd(),
  "data/snapshots/jleagueStandings.html"
);

if (!fs.existsSync(filePath)) {
  console.error("Snapshot not found. Run: node scripts/save-source-snapshots.mjs");
  process.exit(1);
}

const html = fs.readFileSync(filePath, "utf8");
const $ = cheerio.load(html);

console.log("=== Tables ===");
$("table").each((i, table) => {
  const rows = $(table).find("tr").length;
  const text = $(table).text().replace(/\s+/g, " ").trim().slice(0, 300);

  console.log(`[table ${i}] rows=${rows}`);
  console.log(text);
  console.log("");
});

console.log("=== Rows containing 鹿島 ===");
$("tr").each((i, row) => {
  const text = $(row).text().replace(/\s+/g, " ").trim();

  if (text.includes("鹿島")) {
    console.log(`[tr ${i}]`);
    console.log(text);
    console.log("");

    $(row)
      .find("th, td")
      .each((cellIndex, cell) => {
        console.log(`  [cell ${cellIndex}]`);
        console.log(`  text: ${$(cell).text().replace(/\s+/g, " ").trim()}`);
        console.log(`  html: ${$(cell).html()?.slice(0, 500)}`);
        console.log("");
      });
  }
});