export async function fetchHtml(url: string) {
  try {
    const response = await fetch(url, {
      cache: "no-store",
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        Accept:
          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      },
    });

    if (!response.ok) {
      console.warn(`[fetchHtml] Failed: ${response.status} ${url}`);
      return "";
    }

    return response.text();
  } catch (error) {
    console.warn("[fetchHtml] Error:", error);
    return "";
  }
}