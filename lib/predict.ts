import type { Outcome, Prediction, Reason, TeamStats, TotoMatch } from "@/types/toto";

const outcomeLabel: Record<Outcome, string> = { home: "ホーム勝ち", draw: "引分", away: "アウェイ勝ち" };

function formPoints(form: string): number {
  return form.split("").reduce((sum, r) => sum + (r === "W" ? 3 : r === "D" ? 1 : 0), 0);
}

function teamBase(team: TeamStats, opponent: TeamStats, side: "home" | "away"): number {
  const rankScore = (opponent.rank - team.rank) * 1.15;
  const formScore = (formPoints(team.last5) - formPoints(opponent.last5)) * 1.1;
  const gdScore = ((team.goalsFor - team.goalsAgainst) - (opponent.goalsFor - opponent.goalsAgainst)) * 0.35;
  const venueRate = side === "home" ? team.homeWinRate ?? 0.4 : team.awayWinRate ?? 0.3;
  const venueScore = (venueRate - 0.35) * 22;
  const restScore = (team.restDays - opponent.restDays) * 1.4;
  const injuryScore = (opponent.injuryLevel - team.injuryLevel) * 2.2;
  const homeBonus = side === "home" ? 5.5 : -1.5;
  return 50 + rankScore + formScore + gdScore + venueScore + restScore + injuryScore + homeBonus;
}

function weatherDrawBoost(weather: TotoMatch["weather"]): number {
  if (weather === "強雨") return 8;
  if (weather === "雨") return 5;
  if (weather === "曇") return 1.5;
  return 0;
}

function normalize(homeRaw: number, awayRaw: number, drawRaw: number): Record<Outcome, number> {
  const eH = Math.exp(homeRaw / 18);
  const eA = Math.exp(awayRaw / 18);
  const eD = Math.exp(drawRaw / 18);
  const total = eH + eA + eD;
  return {
    home: Math.round((eH / total) * 100),
    draw: Math.round((eD / total) * 100),
    away: Math.round((eA / total) * 100)
  };
}

function confidence(edge: number, probability: number): Prediction["confidence"] {
  if (edge >= 12 && probability >= 54) return "S";
  if (edge >= 8 && probability >= 48) return "A";
  if (edge >= 4 && probability >= 42) return "B";
  return "C";
}

export function predictMatch(match: TotoMatch): Prediction {
  const homeRaw = teamBase(match.home, match.away, "home");
  const awayRaw = teamBase(match.away, match.home, "away");
  const closeGame = Math.max(0, 16 - Math.abs(homeRaw - awayRaw)) * 0.65;
  const drawRaw = 43 + closeGame + weatherDrawBoost(match.weather) - Math.abs(formPoints(match.home.last5) - formPoints(match.away.last5)) * 0.25;
  const probabilities = normalize(homeRaw, awayRaw, drawRaw);

  const outcomes: Outcome[] = ["home", "draw", "away"];
  const pick = outcomes.reduce((best, o) => probabilities[o] - match.voteRate[o] > probabilities[best] - match.voteRate[best] ? o : best, "home" as Outcome);
  const edge = probabilities[pick] - match.voteRate[pick];

  const reasons: Reason[] = [
    { label: "順位差", value: Math.round((match.away.rank - match.home.rank) * 1.15), note: `${match.home.name}${match.home.rank}位 / ${match.away.name}${match.away.rank}位` },
    { label: "直近5試合", value: Math.round((formPoints(match.home.last5) - formPoints(match.away.last5)) * 1.1), note: `${match.home.last5} vs ${match.away.last5}` },
    { label: "得失点差", value: Math.round(((match.home.goalsFor - match.home.goalsAgainst) - (match.away.goalsFor - match.away.goalsAgainst)) * 0.35), note: `ホーム${match.home.goalsFor - match.home.goalsAgainst} / アウェイ${match.away.goalsFor - match.away.goalsAgainst}` },
    { label: "ホーム優位", value: Math.round(((match.home.homeWinRate ?? 0.4) - 0.35) * 22 + 5.5), note: `ホーム勝率${Math.round((match.home.homeWinRate ?? 0) * 100)}%` },
    { label: "日程・疲労", value: Math.round((match.home.restDays - match.away.restDays) * 1.4), note: `休養 ${match.home.restDays}日 vs ${match.away.restDays}日` },
    { label: "主力欠場リスク", value: Math.round((match.away.injuryLevel - match.home.injuryLevel) * 2.2), note: `欠場リスク ${match.home.injuryLevel} vs ${match.away.injuryLevel}` },
    { label: "天候", value: Math.round(-weatherDrawBoost(match.weather) * 0.4), note: `${match.weather}は引分寄りに補正` }
  ];

  return { matchId: match.id, probabilities, pick, confidence: confidence(edge, probabilities[pick]), edge, reasons };
}

export function labelOutcome(outcome: Outcome): string { return outcomeLabel[outcome]; }
export function sign(n: number): string { return n > 0 ? `+${n}` : `${n}`; }
