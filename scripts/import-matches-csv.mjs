import fs from "node:fs";
import path from "node:path";

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SERVICE_KEY) {
  console.error("Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment.");
  process.exit(1);
}

const csvPath = process.argv[2] ?? "data/imports/matches-template.csv";
const csv = fs.readFileSync(path.resolve(csvPath), "utf8").trim();

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  const headers = lines.shift().split(",").map((h) => h.trim());
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((h, i) => [h, values[i]?.trim() ?? ""]));
  });
}

function nullableNumber(v) {
  if (v === "" || v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

const rows = parseCsv(csv);
const roundNo = rows[0]?.round_no;
if (!roundNo) throw new Error("CSV must include round_no.");

async function upsert(table, payload, onConflict) {
  const url = `${SUPABASE_URL}/rest/v1/${table}?on_conflict=${onConflict}`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      apikey: SERVICE_KEY,
      Authorization: `Bearer ${SERVICE_KEY}`,
      "Content-Type": "application/json",
      Prefer: "resolution=merge-duplicates,return=representation",
    },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`${table} upsert failed: ${res.status} ${await res.text()}`);
  return res.json();
}

await upsert("toto_rounds", [{ round_no: roundNo, title: `${roundNo} toto`, status: "imported" }], "round_no");

const matches = rows.map((r) => ({
  round_no: r.round_no,
  match_no: nullableNumber(r.match_no),
  kickoff: r.kickoff || null,
  league: r.league || null,
  venue: r.venue || null,
  home_team: r.home_team,
  away_team: r.away_team,
  weather: r.weather || "未取得",
  home_rank: nullableNumber(r.home_rank),
  away_rank: nullableNumber(r.away_rank),
  home_points: nullableNumber(r.home_points),
  away_points: nullableNumber(r.away_points),
  home_last5: r.home_last5 || null,
  away_last5: r.away_last5 || null,
  home_goals_for: nullableNumber(r.home_goals_for),
  home_goals_against: nullableNumber(r.home_goals_against),
  away_goals_for: nullableNumber(r.away_goals_for),
  away_goals_against: nullableNumber(r.away_goals_against),
  home_win_rate: nullableNumber(r.home_win_rate),
  away_win_rate: nullableNumber(r.away_win_rate),
  home_rest_days: nullableNumber(r.home_rest_days),
  away_rest_days: nullableNumber(r.away_rest_days),
  home_injury_level: nullableNumber(r.home_injury_level),
  away_injury_level: nullableNumber(r.away_injury_level),
  vote_home: nullableNumber(r.vote_home),
  vote_draw: nullableNumber(r.vote_draw),
  vote_away: nullableNumber(r.vote_away),
}));

await upsert("matches", matches, "round_no,match_no");
console.log(`Imported ${matches.length} matches for ${roundNo}.`);
