export type SupabaseConfig = {
  url: string;
  key: string;
};

export function getSupabaseConfig(): SupabaseConfig | null {
  const url =
    process.env.NEXT_PUBLIC_SUPABASE_URL ??
    process.env.SUPABASE_URL;

  const key =
    process.env.SUPABASE_SERVICE_ROLE_KEY ??
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!url || !key) return null;

  return { url, key };
}

export async function supabaseSelect<T>(
  table: string,
  query = "select=*"
): Promise<T[]> {
  const config = getSupabaseConfig();

  if (!config) {
    console.warn("[supabaseSelect] Supabase environment variables are not set.");
    return [];
  }

  const endpoint = `${config.url}/rest/v1/${table}?${query}`;

  const res = await fetch(endpoint, {
    headers: {
      apikey: config.key,
      Authorization: `Bearer ${config.key}`,
    },
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text();
    console.warn(`[supabaseSelect] Failed: ${res.status} ${text}`);
    return [];
  }

  return res.json();
}


export async function supabaseUpsert<T>(
  table: string,
  rows: T[],
  onConflict?: string
): Promise<boolean> {
  const config = getSupabaseConfig();

  if (!config) {
    console.warn("[supabaseUpsert] Supabase environment variables are not set.");
    return false;
  }

  if (rows.length === 0) return true;

  const query = onConflict ? `?on_conflict=${encodeURIComponent(onConflict)}` : "";
  const endpoint = `${config.url}/rest/v1/${table}${query}`;

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      apikey: config.key,
      Authorization: `Bearer ${config.key}`,
      "Content-Type": "application/json",
      Prefer: "resolution=merge-duplicates,return=minimal",
    },
    body: JSON.stringify(rows),
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text();
    console.warn(`[supabaseUpsert] Failed: ${res.status} ${text}`);
    return false;
  }

  return true;
}
