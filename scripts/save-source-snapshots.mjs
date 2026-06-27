import fs from "fs";
import path from "path";

const sources = {
  rakutenSchedule: "https://toto.rakuten.co.jp/toto/schedule/",
  yahooSchedule: "https://toto.yahoo.co.jp/schedule/toto",
  jleagueMatches: "https://www.jleague.jp/match/",
  jleagueStandings: "https://www.jleague.jp/standings/j1/",
};

const outputDir = path.join(process.cwd(), "data/snapshots");

fs.mkdirSync(outputDir, { recursive: true });

for (const [key, url] of Object.entries(sources)) {
  const res = await fetch(url, {
    headers: {
      "User-Agent": "Mozilla/5.0",
    },
  });

  if (!res.ok) {
    console.error(`${key}: failed ${res.status}`);
    continue;
  }

  const html = await res.text();
  fs.writeFileSync(path.join(outputDir, `${key}.html`), html, "utf8");
  console.log(`${key}: saved ${html.length} chars`);
}