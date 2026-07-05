import type { MatchOutcome } from "@/types/prediction";

export function mapTotoResultToOutcome(result: string): MatchOutcome | null {
  if (result === "1") return "HOME";
  if (result === "0") return "DRAW";
  if (result === "2") return "AWAY";
  return null;
}