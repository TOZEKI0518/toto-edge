import { NextResponse } from "next/server";
import { saveCurrentStandingsSnapshot } from "@/services/standingsHistoryService";

export const dynamic = "force-dynamic";

function isAuthorized(token: string | null) {
  if (!process.env.ADMIN_API_TOKEN) return true;
  return token === process.env.ADMIN_API_TOKEN;
}

function toNumber(value: string | null) {
  if (!value) return undefined;
  const numberValue = Number(value);
  return Number.isInteger(numberValue) ? numberValue : undefined;
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);

    const token = searchParams.get("token");
    const roundNo = toNumber(searchParams.get("round"));
    const snapshotDate = searchParams.get("snapshotDate") ?? undefined;

    if (!isAuthorized(token)) {
      return NextResponse.json(
        { ok: false, error: "Unauthorized." },
        { status: 401 }
      );
    }

    const result = await saveCurrentStandingsSnapshot(roundNo, snapshotDate);

    return NextResponse.json({
      ok: true,
      ...result,
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Snapshot save failed.",
      },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => ({}));

    const token =
      typeof body.token === "string" ? body.token : null;

    if (!isAuthorized(token)) {
      return NextResponse.json(
        { ok: false, error: "Unauthorized." },
        { status: 401 }
      );
    }

    const roundNo =
      typeof body.roundNo === "number" && Number.isInteger(body.roundNo)
        ? body.roundNo
        : undefined;

    const snapshotDate =
      typeof body.snapshotDate === "string" ? body.snapshotDate : undefined;

    const result = await saveCurrentStandingsSnapshot(roundNo, snapshotDate);

    return NextResponse.json({
      ok: true,
      ...result,
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Snapshot save failed.",
      },
      { status: 500 }
    );
  }
}
