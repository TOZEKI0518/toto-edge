// app/api/admin/export-training/route.ts

import { NextResponse } from "next/server";
import { buildDateCutStandings } from "@/services/dateCutStandingService";
import { buildPredictionInputs } from "@/services/predictionInputService";
import { buildFeatureVectors } from "@/services/featureFactory";
import {
  featureVectorsToCsv,
  buildDatasetSummary,
} from "@/services/datasetBuilder";
import { loadRoundFixtures } from "@/services/totoFixtureCacheService";
import { isJLeagueFixture, withLeague } from "@/services/leagueClassifier";
import { mapTotoResultToOutcome } from "@/services/backtestOutcomeMapper";
import {
  buildKnownRoundRange,
  TOTO_BACKTEST_ROUNDS,
} from "@/services/totoRoundCalendar";
import type { MatchResult } from "@/types/feature";

export const dynamic = "force-dynamic";

function isAuthorized(token: string | null) {
  if (!process.env.ADMIN_API_TOKEN) return true;
  return token === process.env.ADMIN_API_TOKEN;
}

function toMatchResult(value: string | null | undefined): MatchResult | undefined {
  const outcome = mapTotoResultToOutcome(value ?? "");

  if (outcome === "HOME") return "H";
  if (outcome === "DRAW") return "D";
  if (outcome === "AWAY") return "A";

  return undefined;
}

function buildRoundNumbers(searchParams: URLSearchParams): number[] {
  const from = searchParams.get("from");
  const to = searchParams.get("to");
  const rawLimit = Number(searchParams.get("limit") ?? "20");
  const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? rawLimit : 20;

  if (from || to) {
    return buildKnownRoundRange(from ?? undefined, to ?? undefined).map(Number);
  }

  return TOTO_BACKTEST_ROUNDS.slice(0, limit).map(Number);
}

function shouldUseJLeagueOnly(searchParams: URLSearchParams) {
  const league = searchParams.get("league") ?? searchParams.get("target");

  if (!league) return false;

  return ["j", "jleague", "j-league", "j.league"].includes(
    league.toLowerCase()
  );
}

async function buildRoundFeatures(roundNo: number, jLeagueOnly: boolean) {
  const standingResult = await buildDateCutStandings(roundNo);

  if (!standingResult.targetDate || standingResult.standings.length === 0) {
    return {
      features: [],
      debug: {
        roundNo,
        targetDate: standingResult.targetDate ?? null,
        standings: standingResult.standings.length,
        fixtures: 0,
        selectedFixtures: 0,
        inputs: 0,
        features: 0,
        withResult: 0,
        skippedReason: "No targetDate or standings.",
      },
    };
  }

  const loadedFixtures = await loadRoundFixtures(roundNo);

  const allFixtures = loadedFixtures.fixtures.map(withLeague);

  const fixtures = jLeagueOnly
    ? allFixtures.filter(isJLeagueFixture)
    : allFixtures;

  const inputs = buildPredictionInputs(fixtures, standingResult.standings);

  const resultByMatchNo = new Map<number, MatchResult>();

  for (const fixture of fixtures) {
    const result = toMatchResult(fixture.totoResult);
    if (result) {
      resultByMatchNo.set(fixture.matchNo, result);
    }
  }

  const features = buildFeatureVectors({
    roundNo,
    snapshotDate: standingResult.targetDate,
    inputs,
    resultByMatchNo,
  });

  return {
    features,
    debug: {
      roundNo,
      targetDate: standingResult.targetDate,
      standings: standingResult.standings.length,
      fixtures: loadedFixtures.fixtures.length,
      selectedFixtures: fixtures.length,
      inputs: inputs.length,
      features: features.length,
      withResult: features.filter((feature) => feature.result).length,
    },
  };
}

async function buildTrainingFeatures(roundNumbers: number[], jLeagueOnly: boolean) {
  const allFeatures = [];
  const debug = [];

  for (const roundNo of roundNumbers) {
    const result = await buildRoundFeatures(roundNo, jLeagueOnly);
    allFeatures.push(...result.features);
    debug.push(result.debug);
  }

  return {
    features: allFeatures,
    debug,
  };
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

    const format = searchParams.get("format") ?? "csv";
    const roundNumbers = buildRoundNumbers(searchParams);
    const jLeagueOnly = shouldUseJLeagueOnly(searchParams);

    const { features, debug } = await buildTrainingFeatures(
      roundNumbers,
      jLeagueOnly
    );

    if (format === "json") {
      return NextResponse.json({
        ok: true,
        mode: jLeagueOnly ? "J.League only" : "All fixtures",
        rounds: roundNumbers,
        summary: buildDatasetSummary(features),
        debug,
        features,
      });
    }

    const csv = featureVectorsToCsv(features);

    return new Response(csv, {
      status: 200,
      headers: {
        "Content-Type": "text/csv; charset=utf-8",
        "Content-Disposition": `attachment; filename="toto-training-dataset.csv"`,
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error:
          error instanceof Error
            ? error.message
            : "Training dataset export failed.",
      },
      { status: 500 }
    );
  }
}