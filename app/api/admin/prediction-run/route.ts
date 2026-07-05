import { NextResponse } from "next/server";
import { savePredictionHistoryForRound } from "@/services/predictionHistoryService";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);

  const round = searchParams.get("round");
  const token = searchParams.get("token");

  if (!process.env.ADMIN_API_TOKEN) {
    return NextResponse.json(
      { ok: false, error: "ADMIN_API_TOKEN is not set." },
      { status: 500 }
    );
  }

  if (token !== process.env.ADMIN_API_TOKEN) {
    return NextResponse.json(
      { ok: false, error: "Unauthorized." },
      { status: 401 }
    );
  }

  if (!round || !Number.isInteger(Number(round))) {
    return NextResponse.json(
      { ok: false, error: "round must be an integer." },
      { status: 400 }
    );
  }

  const result = await savePredictionHistoryForRound(round);

  return NextResponse.json({
    ok: true,
    ...result,
  });
}