import { normalizeMatches } from "@/lib/normalizers/normalizeMatch";
import type { RawTotoMatch } from "@/types/rawToto";

const rawMatches: RawTotoMatch[] = [
  {
    "totoRound": "第XXXX回",
    "matchNo": "1",
    "kickoffAt": "2026-07-04T19:00:00+09:00",
    "venue": "カシマスタジアム",
    "homeShortName": "鹿島",
    "awayShortName": "神戸",
    "homeRank": "2",
    "awayRank": "6",
    "homeRecentPoints": "12",
    "awayRecentPoints": "7",
    "homeGoalDiff": "15",
    "awayGoalDiff": "5",
    "homeWinRate": "0.72",
    "awayWinRate": "0.39",
    "voteHome": "45",
    "voteDraw": "26",
    "voteAway": "29",
    "weather": "曇り"
  },
  {
    "totoRound": "第XXXX回",
    "matchNo": "2",
    "kickoffAt": "2026-07-04T19:00:00+09:00",
    "venue": "日産スタジアム",
    "homeShortName": "横浜FM",
    "awayShortName": "川崎",
    "homeRank": "9",
    "awayRank": "8",
    "homeRecentPoints": "6",
    "awayRecentPoints": "8",
    "homeGoalDiff": "0",
    "awayGoalDiff": "3",
    "homeWinRate": "0.45",
    "awayWinRate": "0.42",
    "voteHome": "34",
    "voteDraw": "31",
    "voteAway": "35",
    "weather": "雨"
  },
  {
    "totoRound": "第XXXX回",
    "matchNo": "3",
    "kickoffAt": "2026-07-05T18:30:00+09:00",
    "venue": "埼玉スタジアム",
    "homeShortName": "浦和",
    "awayShortName": "FC東京",
    "homeRank": "4",
    "awayRank": "12",
    "homeRecentPoints": "10",
    "awayRecentPoints": "4",
    "homeGoalDiff": "10",
    "awayGoalDiff": "-9",
    "homeWinRate": "0.67",
    "awayWinRate": "0.22",
    "voteHome": "49",
    "voteDraw": "27",
    "voteAway": "24",
    "weather": "晴れ"
  }
];

export const currentMatches = normalizeMatches(rawMatches);
