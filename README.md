# TOTO EDGE

期待値で選ぶ toto 分析MVPです。AI予想確率と投票率の差分から「狙い目」を表示します。

## 起動

```powershell
cd "C:\Users\81805\TOTO EDGE"
npm install
npm run dev
```

http://localhost:3000 を開きます。

## Ver0.2: データ取得エンジン + Supabase連携

今回追加したもの:

- Supabase用DBスキーマ: `supabase/schema.sql`
- CSV投入テンプレート: `data/imports/matches-template.csv`
- CSV→Supabase投入スクリプト: `scripts/import-matches-csv.mjs`
- toto投票率ページ調査用スクリプト: `scripts/fetch-toto-votes.mjs`
- Supabase未設定時はサンプルデータに自動フォールバック

## Supabase設定手順

### 1. Supabaseでテーブル作成

SupabaseのSQL Editorで `supabase/schema.sql` を実行します。

### 2. 環境変数を作成

`.env.example` をコピーして `.env.local` を作ります。

```powershell
copy .env.example .env.local
```

`.env.local` に以下を入れます。

```env
NEXT_PUBLIC_SUPABASE_URL=https://xxxxxxxxxxxxxxxxxxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=xxxxx
SUPABASE_SERVICE_ROLE_KEY=xxxxx
```

注意: `SUPABASE_SERVICE_ROLE_KEY` はブラウザに出してはいけません。ローカルの投入スクリプトだけで使います。

### 3. CSVを投入

まずはテンプレートを使って1試合分を投入できます。

```powershell
npm run import:matches -- data/imports/matches-template.csv
```

13試合分に増やす場合は、`data/imports/matches-template.csv` をコピーして編集してください。

### 4. 起動

```powershell
npm run dev
```

トップ画面の「モード」が `Supabase連携` になれば成功です。

## データ取得方針

MVPでは完全自動スクレイピングよりも、まずは「無料で安定して運用できる」ことを優先します。

1. toto公式/楽天toto/Yahoo! totoで開催回・対戦カード・投票率を確認
2. Jリーグ公式データサイト等で順位・成績を確認
3. CSVに入力
4. Supabaseへ投入
5. アプリが自動で予想を再計算

この流れで精度検証を始め、安定した項目から順番に自動取得へ切り替えます。

## 投票率調査用コマンド

```powershell
npm run fetch:toto-votes -- 1636
```

指定回の公開投票率ページのテキストを表示します。最初は表示結果を見ながらCSVへ転記してください。HTML構造が安定していることを確認できたら、次のバージョンで自動パース化します。
