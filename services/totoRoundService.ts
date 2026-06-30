import { fetchHtml } from "@/services/totoFetcher";
import { parseTotoFixturesFromHtml } from "@/services/totoFixtureParser";
import { getLatestYahooTotoRound } from "@/services/yahooTotoService";
import type { TotoRoundData } from "@/types/round";

function getRoundNumber(round: string): string {
  return round.replace(/[^\d]/g, "");
}

export async function getCurrentRoundData(): Promise<TotoRoundData> {
  const latestRound = await getLatestYahooTotoRound();

  const round = latestRound?.round ?? "第1637回";
  const roundNumber = getRoundNumber(round);

  const url = `https://store.toto-dream.com/dcs/subos/screen/pi04/spin011/PGSPIN01101LnkHoldCntLotResultLsttoto.form?popupDispDiv=disp&holdCntId=${roundNumber}`;

  const html = await fetchHtml(url);
  const fixtures = parseTotoFixturesFromHtml(html);

  return {
    round,
    salesStart: latestRound?.salesStart,
    salesEnd: latestRound?.salesEnd,
    fixtures,
  };
}