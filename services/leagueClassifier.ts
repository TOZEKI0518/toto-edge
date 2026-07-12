import { normalizeTeamName } from "@/lib/teamNameNormalizer";
import type { TotoFixture } from "@/types/fixture";

export type LeagueKind =
  | "J.League"
  | "World Cup"
  | "National Team"
  | "Cup/Other"
  | "Unknown";

const J_LEAGUE_TEAMS = new Set(
  [
    // J1 / J2 / J3 でtotoに出やすい短縮名・正式名を両方許容
    "鹿島アントラーズ",
    "浦和レッズ",
    "柏レイソル",
    "FC東京",
    "東京ヴェルディ",
    "FC町田ゼルビア",
    "町田ゼルビア",
    "町田",
    "川崎フロンターレ",
    "横浜F・マリノス",
    "横浜FC",
    "湘南ベルマーレ",
    "アルビレックス新潟",
    "清水エスパルス",
    "名古屋グランパス",
    "京都サンガF.C.",
    "ガンバ大阪",
    "セレッソ大阪",
    "ヴィッセル神戸",
    "ファジアーノ岡山",
    "サンフレッチェ広島",
    "アビスパ福岡",
    "北海道コンサドーレ札幌",
    "ベガルタ仙台",
    "ブラウブリッツ秋田",
    "モンテディオ山形",
    "いわきFC",
    "水戸ホーリーホック",
    "RB大宮アルディージャ",
    "大宮アルディージャ",
    "大宮",
    "ジェフユナイテッド千葉",
    "ヴァンフォーレ甲府",
    "カターレ富山",
    "富山",
    "ジュビロ磐田",
    "藤枝MYFC",
    "レノファ山口FC",
    "山口",
    "徳島ヴォルティス",
    "愛媛FC",
    "FC今治",
    "サガン鳥栖",
    "V・ファーレン長崎",
    "ロアッソ熊本",
    "熊本",
    "大分トリニータ",
    "ヴァンラーレ八戸",
    "八戸",
    "福島ユナイテッドFC",
    "福島",
    "栃木SC",
    "栃木シティ",
    "ザスパ群馬",
    "群馬",
    "SC相模原",
    "相模原",
    "松本山雅FC",
    "松本",
    "AC長野パルセイロ",
    "長野",
    "ツエーゲン金沢",
    "金沢",
    "アスルクラロ沼津",
    "沼津",
    "FC岐阜",
    "岐阜",
    "FC大阪",
    "奈良クラブ",
    "奈良",
    "ガイナーレ鳥取",
    "鳥取",
    "カマタマーレ讃岐",
    "讃岐",
    "高知ユナイテッドSC",
    "高知",
    "ギラヴァンツ北九州",
    "北九州",
    "テゲバジャーロ宮崎",
    "宮崎",
    "鹿児島ユナイテッドFC",
    "鹿児島",
    "FC琉球",
  ].map(normalizeTeamName)
);

const NATIONAL_TEAMS = new Set(
  [
    "日本",
    "韓国",
    "中国",
    "北朝鮮",
    "オーストラリア",
    "イラン",
    "イラク",
    "サウジアラビア",
    "カタール",
    "UAE",
    "アラブ首長国連邦",
    "タイ",
    "ベトナム",
    "インドネシア",
    "ドイツ",
    "フランス",
    "スペイン",
    "イングランド",
    "イタリア",
    "オランダ",
    "ベルギー",
    "ポルトガル",
    "クロアチア",
    "スイス",
    "デンマーク",
    "ポーランド",
    "ブラジル",
    "アルゼンチン",
    "ウルグアイ",
    "チリ",
    "コロンビア",
    "エクアドル",
    "メキシコ",
    "アメリカ",
    "カナダ",
    "コスタリカ",
    "モロッコ",
    "セネガル",
    "ガーナ",
    "カメルーン",
    "ナイジェリア",
    "チュニジア",
    "南アフリカ",
    "ニュージーランド",
  ].map(normalizeTeamName)
);

export function isJLeagueTeam(teamName: string) {
  return J_LEAGUE_TEAMS.has(normalizeTeamName(teamName));
}

export function isNationalTeam(teamName: string) {
  return NATIONAL_TEAMS.has(normalizeTeamName(teamName));
}

export function classifyFixtureLeague(fixture: Pick<TotoFixture, "homeTeam" | "awayTeam">): LeagueKind {
  const homeIsJ = isJLeagueTeam(fixture.homeTeam);
  const awayIsJ = isJLeagueTeam(fixture.awayTeam);
  if (homeIsJ && awayIsJ) return "J.League";

  const homeIsNational = isNationalTeam(fixture.homeTeam);
  const awayIsNational = isNationalTeam(fixture.awayTeam);
  if (homeIsNational && awayIsNational) return "World Cup";

  if (homeIsJ || awayIsJ) return "Cup/Other";

  if (homeIsNational || awayIsNational) return "National Team";

  return "Unknown";
}

export function isJLeagueFixture(fixture: Pick<TotoFixture, "homeTeam" | "awayTeam" | "league">) {
  return (fixture.league ?? classifyFixtureLeague(fixture)) === "J.League";
}

export function withLeague<T extends Pick<TotoFixture, "homeTeam" | "awayTeam">>(fixture: T): T & { league: LeagueKind } {
  return {
    ...fixture,
    league: classifyFixtureLeague(fixture),
  };
}

export function leagueBadgeClass(league: string | undefined) {
  if (league === "J.League") return "bg-cyan-400/15 text-cyan-200";
  if (league === "World Cup") return "bg-red-400/15 text-red-200";
  if (league === "National Team") return "bg-orange-400/15 text-orange-200";
  if (league === "Cup/Other") return "bg-violet-400/15 text-violet-200";
  return "bg-white/10 text-white/60";
}

export function leagueBadgeLabel(league: string | undefined) {
  if (league === "J.League") return "🔵 J.League";
  if (league === "World Cup") return "🔴 World Cup";
  if (league === "National Team") return "🟠 National";
  if (league === "Cup/Other") return "🟣 Cup/Other";
  return "⚪ Unknown";
}
