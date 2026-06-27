# TOTO AI Ver1.0 設計書

作成日: 2026-06-27

## 1. コンセプト

TOTO AI Ver1.0 は、toto の13試合を対象に、単純な勝敗予想ではなく「AI予想確率」と「投票率」のズレから期待値の高い選択肢を探す分析アプリとする。

目的は、ギャンブル的な勘ではなく、以下を可視化すること。

- どちらが勝ちやすいか
- なぜその予想になったか
- 世間の投票率と比べて割安か
- 過去に同じロジックならどれくらい当たったか

## 2. Ver1.0 のゴール

Ver1.0 は「データ自動取得前の実用MVP」とする。

最初からすべてをAPI連携しようとすると詰まりやすいため、まずは sample data / 手入力 / 半自動取り込みで画面とロジックを完成させる。

### Ver1.0で実装する機能

1. 開催回ごとの13試合一覧
2. 各試合のAI予想確率
3. 1 / 0 / 2 の推奨表示
4. 予想理由の表示
5. 投票率との比較
6. 期待値ランキング
7. 信頼度ランク
8. 結果入力後の的中判定
9. 簡易バックテスト用の履歴保存設計

## 3. 予想対象

### totoの記号

| 記号 | 意味 |
|---|---|
| 1 | ホームチームの90分勝利 |
| 0 | その他 / 引分 |
| 2 | ホームチームの90分敗北 |

アプリ上は初心者にも分かりやすく、以下のように表示する。

- 1 = ホーム勝ち
- 0 = 引分
- 2 = アウェイ勝ち

## 4. 画面設計

### 4.1 トップ画面

表示内容:

- アプリ名: TOTO AI
- 開催回: 第XXXX回 toto
- 期待値ランキング
- 13試合カード一覧

カード例:

```text
鹿島 vs 川崎

AI予想
1 ホーム勝ち 58%
0 引分       24%
2 アウェイ   18%

おすすめ: 1 ホーム勝ち
信頼度: A
投票率差分: +16pt

[詳細を見る]
```

### 4.2 詳細画面 / モーダル

表示内容:

- チーム名
- AI予想確率
- 投票率
- 差分
- スコア内訳
- プラス要因
- マイナス要因
- 最終推奨

例:

```text
鹿島 vs 川崎

AI予想
1: 58%
0: 24%
2: 18%

投票率
1: 42%
0: 28%
2: 30%

期待値差分
1: +16pt
0: -4pt
2: -12pt

理由
+ ホーム成績が良い
+ 直近5試合で得点力が高い
+ 相手が中2日
- 雨予報
- 主力CB欠場
```

### 4.3 バックテスト画面

Ver1.0ではUIだけ用意し、後で履歴データを入れられる構造にする。

表示内容:

- 対象開催回数
- 予想試合数
- 的中率
- 平均的中数 / 13
- 最高的中数
- 期待値プラス判定の的中率

## 5. データ設計

### 5.1 matches

試合ごとの基本データ。

| カラム | 型 | 内容 |
|---|---|---|
| id | uuid | 主キー |
| round_id | text | 開催回 |
| match_no | integer | totoの試合番号 |
| date | date | 試合日 |
| home_team | text | ホームチーム |
| away_team | text | アウェイチーム |
| venue | text | 会場 |
| home_rank | integer | ホーム順位 |
| away_rank | integer | アウェイ順位 |
| home_recent_form | text | 直近5試合 |
| away_recent_form | text | 直近5試合 |
| home_goals_for | integer | ホーム得点 |
| home_goals_against | integer | ホーム失点 |
| away_goals_for | integer | アウェイ得点 |
| away_goals_against | integer | アウェイ失点 |
| weather | text | 天候 |
| temperature | numeric | 気温 |
| fatigue_note | text | 日程・疲労メモ |
| injury_note | text | 欠場メモ |
| result | text | 1 / 0 / 2 |

### 5.2 predictions

AI予想結果。

| カラム | 型 | 内容 |
|---|---|---|
| id | uuid | 主キー |
| match_id | uuid | matches.id |
| home_prob | numeric | 1の予想確率 |
| draw_prob | numeric | 0の予想確率 |
| away_prob | numeric | 2の予想確率 |
| recommended_pick | text | 1 / 0 / 2 |
| confidence | text | S / A / B / C |
| reason_json | jsonb | 理由の配列 |
| created_at | timestamp | 作成日時 |

### 5.3 vote_rates

toto投票率。

| カラム | 型 | 内容 |
|---|---|---|
| id | uuid | 主キー |
| match_id | uuid | matches.id |
| vote_home | numeric | 1の投票率 |
| vote_draw | numeric | 0の投票率 |
| vote_away | numeric | 2の投票率 |
| source | text | 取得元 |
| captured_at | timestamp | 取得日時 |

### 5.4 backtests

過去検証。

| カラム | 型 | 内容 |
|---|---|---|
| id | uuid | 主キー |
| round_id | text | 開催回 |
| match_count | integer | 対象試合数 |
| hit_count | integer | 的中数 |
| accuracy | numeric | 的中率 |
| average_edge | numeric | 平均期待値差分 |
| created_at | timestamp | 作成日時 |

## 6. 初期スコアリング設計

Ver1.0では機械学習ではなくルールベースで開始する。

理由:

- 説明可能性が高い
- 外れた原因を分析しやすい
- データが少なくても動く
- 後から重み調整しやすい

### 6.1 基本スコア項目

| 要因 | 初期重み | 内容 |
|---|---:|---|
| 直近5試合 | 20 | 勝点換算で勢いを見る |
| ホーム/アウェイ成績 | 15 | ホーム強さ、遠征弱さ |
| 順位差 | 10 | チーム力の基本指標 |
| 得失点差 | 15 | 攻守バランス |
| 日程・疲労 | 10 | 中2日、連戦、遠征 |
| 主力欠場 | 15 | GK, CB, FW, 司令塔を重視 |
| 天候 | 5 | 大雨・強風・猛暑など |
| モチベーション | 5 | 優勝争い・残留争い |
| 過去対戦 | 5 | 参考程度 |

合計100点ではなく、各チームの相対スコアを作り、softmax的に確率へ変換する。

### 6.2 直近5試合スコア

```text
勝ち = +3
引分 = +1
負け = 0
```

例:

```text
WWDLW = 3+3+1+0+3 = 10点
```

### 6.3 順位差スコア

```text
away_rank - home_rank
```

ホームが上位ならプラス、下位ならマイナス。

### 6.4 得失点差スコア

```text
(home_goals_for - home_goals_against) - (away_goals_for - away_goals_against)
```

### 6.5 引分確率

サッカーは引分が重要なので、単純な2択にしない。

引分確率が上がる条件:

- チーム力が近い
- 互いに得点力が低い
- 雨・強風
- 連戦で疲労
- 両チームとも守備型

初期式:

```text
draw_base = 24
if score_gap < 8: +6
if both_low_scoring: +4
if bad_weather: +3
if both_fatigued: +2
```

## 7. 期待値判定

### 7.1 基本式

```text
edge = AI予想確率 - 投票率
```

例:

```text
AI 58%
投票率 42%
edge +16pt
```

### 7.2 ランク

| edge | 判定 |
|---:|---|
| +15pt以上 | S 狙い目 |
| +10pt以上 | A 良い |
| +5pt以上 | B やや良い |
| 0〜+5pt | C 普通 |
| 0未満 | 見送り |

## 8. 信頼度設計

信頼度は「勝ちやすさ」ではなく「予想の強さ」とする。

| 条件 | ランク |
|---|---|
| AI最大確率60%以上、edge+10pt以上 | S |
| AI最大確率55%以上、edge+7pt以上 | A |
| AI最大確率50%以上 | B |
| それ以外 | C |

## 9. データ取得方針

### 9.1 toto公式データ

- 開催回
- 対象試合
- 投票率
- くじ結果
- 払戻結果

公式サイトから確認できる情報を優先する。

### 9.2 Jリーグデータ

- 日程
- 結果
- 順位
- ニュース
- クラブ情報

最初は手入力またはCSVで十分。

### 9.3 天候

OpenWeatherなどの無料APIで試合会場の天候を取得する想定。

Ver1.0では天候は手入力でも良い。

### 9.4 怪我・欠場情報

完全自動化は難しいため、Ver1.0では手入力メモ。

将来的にはニュース検索・チーム公式情報をAI要約する。

## 10. Next.js構成

```text
toto/
├── app/
│   ├── page.tsx
│   ├── layout.tsx
│   └── globals.css
├── components/
│   ├── MatchCard.tsx
│   ├── MatchDetail.tsx
│   ├── EdgeRanking.tsx
│   ├── PredictionBadge.tsx
│   └── BacktestSummary.tsx
├── data/
│   └── sampleMatches.ts
├── lib/
│   ├── prediction.ts
│   ├── scoring.ts
│   ├── edge.ts
│   └── utils.ts
├── types/
│   └── toto.ts
├── public/
├── package.json
└── README.md
```

## 11. 主要TypeScript型

```ts
export type TotoPick = '1' | '0' | '2';

export type Match = {
  id: string;
  roundId: string;
  matchNo: number;
  date: string;
  homeTeam: string;
  awayTeam: string;
  venue: string;
  homeRank: number;
  awayRank: number;
  homeRecentForm: string;
  awayRecentForm: string;
  homeGoalsFor: number;
  homeGoalsAgainst: number;
  awayGoalsFor: number;
  awayGoalsAgainst: number;
  weather?: string;
  fatigueNote?: string;
  injuryNote?: string;
  voteHome: number;
  voteDraw: number;
  voteAway: number;
  result?: TotoPick;
};

export type Prediction = {
  matchId: string;
  homeProb: number;
  drawProb: number;
  awayProb: number;
  recommendedPick: TotoPick;
  confidence: 'S' | 'A' | 'B' | 'C';
  reasons: PredictionReason[];
};

export type PredictionReason = {
  label: string;
  score: number;
  type: 'positive' | 'negative' | 'neutral';
};
```

## 12. Ver1.0のサンプルロジック

```ts
export function calculatePrediction(match: Match): Prediction {
  const homeForm = formScore(match.homeRecentForm);
  const awayForm = formScore(match.awayRecentForm);

  const rankScore = match.awayRank - match.homeRank;
  const goalDiffScore =
    (match.homeGoalsFor - match.homeGoalsAgainst) -
    (match.awayGoalsFor - match.awayGoalsAgainst);

  let homeRaw = 50;
  let awayRaw = 50;
  let drawRaw = 24;

  homeRaw += (homeForm - awayForm) * 1.8;
  homeRaw += rankScore * 1.1;
  homeRaw += goalDiffScore * 0.7;
  homeRaw += 5; // home advantage

  awayRaw -= (homeForm - awayForm) * 1.8;
  awayRaw -= rankScore * 1.1;
  awayRaw -= goalDiffScore * 0.7;

  const scoreGap = Math.abs(homeRaw - awayRaw);
  if (scoreGap < 8) drawRaw += 6;
  if (match.weather?.includes('雨')) drawRaw += 3;

  return normalizeToPrediction(homeRaw, drawRaw, awayRaw, match);
}
```

## 13. 開発ステップ

### Step 1: 画面MVP

- sampleMatches.tsを作成
- 13試合カード表示
- 詳細モーダル表示
- 期待値ランキング表示

### Step 2: ロジックMVP

- formScore
- rankScore
- goalDiffScore
- draw adjustment
- edge calculation
- confidence calculation

### Step 3: 結果入力

- resultを入力
- 推奨が当たったか表示
- 的中数集計

### Step 4: Supabase連携

- rounds
- matches
- predictions
- vote_rates
- backtests

### Step 5: データ半自動化

- toto公式投票率の取り込み補助
- Jリーグ順位・成績のCSV取り込み
- 天候API連携

## 14. 注意点

このアプリは投資・ギャンブルの利益を保証するものではない。

公営くじには控除率があり、長期的に必ず儲かるものではない。アプリ上にも以下の注意書きを表示する。

```text
本アプリは分析・学習目的の予想支援ツールです。購入判断は自己責任で行ってください。利益や的中を保証するものではありません。
```

## 15. 将来構想

### Ver2

- Supabase保存
- バックテスト画面
- 開催回履歴
- 的中率グラフ

### Ver3

- 自動データ取得
- 天候API連携
- ニュースAI要約
- 欠場情報メモ

### Ver4

- 重み自動調整
- 期待値モデル改善
- 回収率風シミュレーション

### Ver5

- 競輪・競馬への横展開
- スポーツ予想AIエンジン化

## 16. 最初に作るべき完成形

最初の完成イメージは以下。

```text
TOTO AI
第XXXX回 toto

期待値ランキング
1位 鹿島勝ち +16pt S
2位 神戸勝ち +12pt A
3位 広島引分 +9pt B

13試合一覧
各試合カード + 詳細理由
```

これで「使って楽しい」「改善したくなる」MVPになる。
