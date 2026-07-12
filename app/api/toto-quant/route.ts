import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";

type CsvRow = Record<string, string>;

function parseCsv(text: string): CsvRow[] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && quoted && next === '"') {
      field += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(field);
      field = "";
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
    } else {
      field += char;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  const [header, ...body] = rows;
  if (!header) return [];

  return body.map((values) =>
    Object.fromEntries(
      header.map((name, index) => [
        name.replace(/^\uFEFF/, "").trim(),
        values[index]?.trim() ?? "",
      ]),
    ),
  );
}

async function readJson(relativePath: string): Promise<unknown | null> {
  try {
    const fullPath = path.join(process.cwd(), relativePath);
    return JSON.parse(await readFile(fullPath, "utf8"));
  } catch {
    return null;
  }
}

async function readCsv(relativePath: string): Promise<CsvRow[]> {
  try {
    const fullPath = path.join(process.cwd(), relativePath);
    return parseCsv(await readFile(fullPath, "utf8"));
  } catch {
    return [];
  }
}

export async function GET() {
  const [predictions, teamFeatures, probability, expectedValue, monteCarlo] =
    await Promise.all([
      readCsv("ml/round_predictions.csv"),
      readCsv("ml/player_engine/data/processed/team_features.csv"),
      readJson("ml/probability_summary.json"),
      readJson("ml/expected_value_summary.json"),
      readJson("ml/monte_carlo_summary.json"),
    ]);

  return NextResponse.json(
    {
      generatedAt: new Date().toISOString(),
      predictions,
      teamFeatures,
      probability,
      expectedValue,
      monteCarlo,
    },
    {
      headers: {
        "Cache-Control": "no-store",
      },
    },
  );
}
