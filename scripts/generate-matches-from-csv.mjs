import fs from "fs";
import path from "path";

const inputPath = path.join(process.cwd(), "data/imports/current-toto-matches.csv");
const outputPath = path.join(process.cwd(), "data/generated/currentMatches.ts");

const text = fs.readFileSync(inputPath, "utf8").trim();
const [headerLine, ...lines] = text.split(/\r?\n/);
const headers = headerLine.split(",");

const rows = lines.map((line) => {
  const values = line.split(",");
  return Object.fromEntries(headers.map((h, i) => [h, values[i]]));
});

const matches = rows.map((r) => ({
  id: `${r.totoRound}-${r.matchNo}`,
  totoRound: r.totoRound,
  matchNo: Number(r.matchNo),
  kickoffAt: r.kickoffAt,
  venue: r.venue,
  homeTeam: {
    id: r.homeShortName,
    name: r.homeShortName,
    shortName: r.homeShortName,
    league: "J1",
  },
  awayTeam: {
    id: r.awayShortName,
    name: r.awayShortName,
    shortName: r.awayShortName,
    league: "J1",
  },
  homeStats: {
    teamId: r.homeShortName,
    rank: Number(r.homeRank),
    points: 0,
    wins: 0,
    draws: 0,
    losses: 0,
    goalsFor: 0,
    goalsAgainst: 0,
    goalDifference: Number(r.homeGoalDiff),
    homeWinRate: Number(r.homeWinRate),
    awayWinRate: 0,
    recentFormPoints: Number(r.homeRecentPoints),
  },
  awayStats: {
    teamId: r.awayShortName,
    rank: Number(r.awayRank),
    points: 0,
    wins: 0,
    draws: 0,
    losses: 0,
    goalsFor: 0,
    goalsAgainst: 0,
    goalDifference: Number(r.awayGoalDiff),
    homeWinRate: 0,
    awayWinRate: Number(r.awayWinRate),
    recentFormPoints: Number(r.awayRecentPoints),
  },
  weather: {
    condition: r.weather,
    temperature: 25,
    windSpeed: 3,
  },
  voteRates: {
    home: Number(r.voteHome),
    draw: Number(r.voteDraw),
    away: Number(r.voteAway),
  },
}));

fs.mkdirSync(path.dirname(outputPath), { recursive: true });

fs.writeFileSync(
  outputPath,
  `import type { TotoMatch } from "@/types/toto";\n\nexport const currentMatches: TotoMatch[] = ${JSON.stringify(matches, null, 2)};\n`,
  "utf8"
);

console.log(`Generated ${matches.length} matches.`);