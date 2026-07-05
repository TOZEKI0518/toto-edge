const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

export type PredictionRun = {
  id: number;
  round_no: number;
  data_snapshot_round: number | null;
  algorithm_version: string;
  hit_count: number | null;
  total_matches: number | null;
  hit_rate: number | null;
  created_at: string;
};

export async function getPredictionRuns(): Promise<PredictionRun[]> {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error("Supabase environment variables are not set.");
  }

  const res = await fetch(
    `${SUPABASE_URL}/rest/v1/prediction_runs?select=*&order=created_at.desc&limit=20`,
    {
      headers: {
        apikey: SUPABASE_SERVICE_ROLE_KEY,
        Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
      },
      cache: "no-store",
    }
  );

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to fetch prediction runs: ${res.status} ${text}`);
  }

  return res.json();
}