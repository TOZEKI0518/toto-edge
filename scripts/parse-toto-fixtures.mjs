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

const table = $("table").eq(2);
const fixtures = [];

table.find("tbody tr").each((_, row) => {
  const cells = $(row)
    .find("th, td")
    .map((_, cell) => $(cell).text().replace(/\s+/g, " ").trim())
    .get()
    .filter(Boolean);

  if (cells.length < 6) return;

  fixtures.push({
    date: cells[0],
    venue: cells[1],
    matchNo: cells[2],
    homeTeam: cells[3],
    result: cells[4],
    awayTeam: cells[5],
    totoResult: cells[6],
  });
});

console.table(fixtures);
console.log(`Parsed ${fixtures.length} fixtures.`);