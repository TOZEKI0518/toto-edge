const TEAM_NAME_MAP: Record<string, string> = {
  "ＦＣ東京": "FC東京",
  "横浜Ｆ・マリノス": "横浜F・マリノス",
  "京都サンガＦ.Ｃ.": "京都サンガF.C.",
  "Ｖ・ファーレン長崎": "V・ファーレン長崎",
};

export function normalizeTeamName(name: string): string {
  const cleaned = name.replace(/\s+/g, "").trim();

  if (TEAM_NAME_MAP[cleaned]) {
    return TEAM_NAME_MAP[cleaned];
  }

  return cleaned
    .replace(/Ｆ/g, "F")
    .replace(/Ｃ/g, "C")
    .replace(/Ｖ/g, "V")
    .replace(/．/g, ".");
}

export function areSameTeam(a: string, b: string): boolean {
  return normalizeTeamName(a) === normalizeTeamName(b);
}
export function removeDuplicatedTeamName(name: string): string {
  const value = name.trim();

  if (
    value.length % 2 === 0 &&
    value.slice(0, value.length / 2) === value.slice(value.length / 2)
  ) {
    return value.slice(0, value.length / 2);
  }

  return value;
}