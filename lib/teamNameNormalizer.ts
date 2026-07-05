const TEAM_NAME_MAP: Record<string, string> = {
  "FC東京": "FC東京",
  "ＦＣ東京": "FC東京",

  "Ｃ大阪": "セレッソ大阪",
  "G大阪": "ガンバ大阪",
  "Ｇ大阪": "ガンバ大阪",

  "東京Ｖ": "東京ヴェルディ",

  "川崎Ｆ": "川崎フロンターレ",

  "横浜FC": "横浜FC",
  "横浜ＦＣ": "横浜FC",

  "千葉": "ジェフユナイテッド千葉",

  "甲府": "ヴァンフォーレ甲府",

  "清水": "清水エスパルス",

  "新潟": "アルビレックス新潟",

  "柏": "柏レイソル",

  "鹿島": "鹿島アントラーズ",

  "名古屋": "名古屋グランパス",

  "福岡": "アビスパ福岡",

  "岡山": "ファジアーノ岡山",

  "湘南": "湘南ベルマーレ",

  "北海道コンサドーレ札幌": "北海道コンサドーレ札幌",
  "コンサドーレ札幌": "北海道コンサドーレ札幌",
  "札幌": "北海道コンサドーレ札幌",

  "ベガルタ仙台": "ベガルタ仙台",
  "仙台": "ベガルタ仙台",

  "山形": "モンテディオ山形",

  "秋田": "ブラウブリッツ秋田",

  "水戸": "水戸ホーリーホック",

  "栃木Ｃ": "栃木シティ",

  "愛媛": "愛媛FC",

  "琉球": "FC琉球",

  "鳥栖": "サガン鳥栖",

  "いわきFC": "いわきFC",
  "いわきＦＣ": "いわきFC",
  "いわき": "いわきFC",

  "浦和": "浦和レッズ",

  "神戸": "ヴィッセル神戸",

  "京都": "京都サンガF.C.",
  
  "長崎": "V・ファーレン長崎",
  
  "横浜FM": "横浜F・マリノス",
  
  "広島": "サンフレッチェ広島",
  
  "ジュビロ磐田": "ジュビロ磐田",
  "磐田": "ジュビロ磐田",
  
  "徳島ヴォルティス": "徳島ヴォルティス",
  "徳島": "徳島ヴォルティス",
  
  "FC今治": "FC今治",
  "ＦＣ今治": "FC今治",
  "今治": "FC今治",
  
  "藤枝MYFC": "藤枝MYFC",
  "藤枝ＭＹＦＣ": "藤枝MYFC",
  "藤枝": "藤枝MYFC",
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