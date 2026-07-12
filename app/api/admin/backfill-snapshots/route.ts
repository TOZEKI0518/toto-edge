import { NextResponse } from "next/server";
import { getJleagueStandings } from "@/services/jleagueStandingService";
import {
  listRoundDates,
  saveStandingsSnapshot,
} from "@/services/standingsHistoryService";

export const dynamic = "force-dynamic";

function isAuthorized(token: string | null) {
  if (!process.env.ADMIN_API_TOKEN) return true;
  return token === process.env.ADMIN_API_TOKEN;
}

function toNumber(value: string | null, fallback: number) {
  if (!value) return fallback;
  const numberValue = Number(value);
  return Number.isInteger(numberValue) && numberValue > 0 ? numberValue : fallback;
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const token = searchParams.get("token");

    if (!isAuthorized(token)) {
      return NextResponse.json(
        { ok: false, error: "Unauthorized." },
        { status: 401 }
      );
    }

    const limit = toNumber(searchParams.get("limit"), 300);
    const fromRound = searchParams.get("from") ? Number(searchParams.get("from")) : null;
    const toRound = searchParams.get("to") ? Number(searchParams.get("to")) : null;

    let roundDates = await listRoundDates(limit);

    if (Number.isInteger(fromRound)) {
      roundDates = roundDates.filter((row) => row.round_no >= Number(fromRound));
    }

    if (Number.isInteger(toRound)) {
      roundDates = roundDates.filter((row) => row.round_no <= Number(toRound));
    }

    const standings = await getJleagueStandings();

    const results = [];
    for (const row of roundDates) {
      const saved = await saveStandingsSnapshot(
        standings,
        row.round_no,
        row.round_date
      );
      results.push(saved);
    }

    return NextResponse.json({
      ok: true,
      source: "current_jleague_standings_copied_to_each_round_date",
      message:
        "toto_round_datesの日付に合わせてstandings_historyへ回号別Snapshotを保存しました。過去順位を復元するものではなく、現在取得できる順位表を各回の日付で保存します。",
      roundCount: results.length,
      savedRows: results.reduce((sum, item) => sum + item.savedCount, 0),
      results,
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error:
          error instanceof Error ? error.message : "Backfill snapshots failed.",
      },
      { status: 500 }
    );
  }
}
