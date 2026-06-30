import type { TotoRoundData } from "@/types/round";

export async function getCurrentRoundData(): Promise<TotoRoundData> {
  return {
    round: "第1638回",
    fixtures: [],
  };
}