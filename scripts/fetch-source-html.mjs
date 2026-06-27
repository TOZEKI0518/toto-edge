import fs from "fs";
import path from "path";

const sources = {
  yahooSchedule: "https://toto.yahoo.co.jp/schedule/toto",
  rakutenSchedule: "https://toto.rakuten.co.jp/toto/schedule/",
  jleagueMatches: "https://www.jleague.jp/match/",
  jleagueStandings: "https://www.jleague.jp/standings/j1/",
};

const outputDir = path.join(process.cwd(), "data/raw-html");

fs.mkdirSync(outputDir, { recursive: true });

for (const [key, url] of Object.entries(sources)) {
  console.log(`Fetching ${key}: ${url}`);

  const res = await fetch(url, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    },
  });

  if (!res.ok) {
    console.error(`Failed: ${key} ${res.status}`);
    continue;
  }

  const html = await res.text();
  const filePath = path.join(outputDir, `${key}.html`);

  fs.writeFileSync(filePath, html, "utf8");

  console.log(`Saved ${filePath}`);
}