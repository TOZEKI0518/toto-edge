export type ParsedTotoPage = {
  title: string;
  textLength: number;
};

export function parseTotoPage(html: string): ParsedTotoPage {
  const titleMatch = html.match(/<title>(.*?)<\/title>/i);

  return {
    title: titleMatch?.[1]?.trim() ?? "No title",
    textLength: html.length,
  };
}