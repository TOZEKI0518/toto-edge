export type SourceFetchResult = {
  sourceKey: string;
  url: string;
  title: string;
  textLength: number;
  fetchedAt: string;
  ok: boolean;
  errorMessage?: string;
};