import fs from "fs";
import path from "path";

const filePath = path.join(process.cwd(), "data/snapshots/yahooSchedule.html");

if (!fs.existsSync(filePath)) {
  console.error("Snapshot not found. Run: node scripts/save-source-snapshots.mjs");
  process.exit(1);
}

const html = fs.readFileSync(filePath, "utf8");

const text = html
  .replace(/<script[\s\S]*?<\/script>/gi, "")
  .replace(/<style[\s\S]*?<\/style>/gi, "")
  .replace(/<[^>]+>/g, " ")
  .replace(/&nbsp;/g, " ")
  .replace(/&amp;/g, "&")
  .replace(/\s+/g, " ")
  .trim();

const regex =
  /(販売中|終了)?\s*第(\d+)回\s*（(\d{4}\/\d{1,2}\/\d{1,2})-(\d{4}\/\d{1,2}\/\d{1,2})）/g;

const rounds = [];

for (const match of text.matchAll(regex)) {
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

console.table(rounds.slice(0, 20));
console.log(`Parsed ${rounds.length} rounds.`);