import { fetchHtml } from "@/services/totoFetcher";

export type TotoRoundInfo = {
  roundNo: string;
  matchDate: string | null;
  isJleague: boolean;
};

function extractDate(html: string): string | null {
  const m = html.match(/(\d{4})\/(\d{1,2})\/(\d{1,2})/);
  if (!m) return null;
  const [, y, mo, d] = m;
  return `${y}-${mo.padStart(2,"0")}-${d.padStart(2,"0")}`;
}

export async function getTotoRoundInfo(roundNo: string): Promise<TotoRoundInfo> {
  const url =
    `https://store.toto-dream.com/dcs/subos/screen/pi04/spin011/PGSPIN01101LnkHoldCntLotResultLsttoto.form?popupDispDiv=disp&holdCntId=${roundNo}`;

  const html = await fetchHtml(url);

  return {
    roundNo,
    matchDate: extractDate(html),
    isJleague: /J1|J2|J3/.test(html),
  };
}
