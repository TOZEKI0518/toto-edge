import type { TotoFixture } from "./fixture";

export type TotoRoundData = {
  round: string;

  salesStart?: string;
  salesEnd?: string;

  fixtures: TotoFixture[];
};