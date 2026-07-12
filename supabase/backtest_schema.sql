create table if not exists standings_history (
  id bigint generated always as identity primary key,
  round_no integer not null,
  snapshot_date date,
  league text,
  team_name text not null,
  rank integer,
  points integer,
  goal_difference integer,
  matches integer,
  wins integer,
  draws integer,
  losses integer,
  goals_for integer,
  goals_against integer,
  created_at timestamptz default now()
);

-- 既存DB向けの安全な追記。古い standings_history に列が無くても壊れないようにする。
alter table standings_history
add column if not exists snapshot_date date;

alter table standings_history
add column if not exists league text;

create index if not exists idx_standing_round
on standings_history(round_no);

create index if not exists idx_standing_team
on standings_history(team_name);

create index if not exists idx_standing_round_team
on standings_history(round_no, team_name);

create index if not exists idx_standing_snapshot_date
on standings_history(snapshot_date);

create table if not exists prediction_history (
  id bigint generated always as identity primary key,
  round_no integer not null,
  match_no integer not null,
  home_team text,
  away_team text,
  predicted_outcome text,
  actual_outcome text,
  probability numeric,
  confidence text,
  total_score numeric,
  hit boolean,
  algorithm_version text,
  created_at timestamptz default now()
);

create index if not exists idx_prediction_round
on prediction_history(round_no);

create table if not exists prediction_runs (
  id bigint generated always as identity primary key,
  round_no integer not null,
  algorithm_version text not null,
  hit_count integer,
  total_matches integer,
  hit_rate numeric,
  created_at timestamptz default now()
);

create index if not exists idx_prediction_runs_round
on prediction_runs(round_no);

create index if not exists idx_prediction_runs_version
on prediction_runs(algorithm_version);

alter table prediction_history
add column if not exists run_id bigint references prediction_runs(id);

create index if not exists idx_prediction_history_run
on prediction_history(run_id);

alter table prediction_runs
add column if not exists data_snapshot_round integer;


create table if not exists public.toto_round_dates (
  round_no integer primary key,
  round_date date not null,
  created_at timestamptz default now()
);

create index if not exists idx_toto_round_dates_round_date
on public.toto_round_dates(round_date);

-- 既存DB向け: standings_history に回号・日付列が無い場合は追加する。
alter table public.standings_history
add column if not exists round_no integer;

alter table public.standings_history
add column if not exists snapshot_date date;

-- Backtest高速化用: 楽天totoの回号別試合結果キャッシュ。
-- 初回取得したHTMLの解析結果を保存し、2回目以降のBacktest/Summaryで外部fetchを減らします。
create table if not exists public.toto_round_fixtures (
  round_no integer not null,
  match_no integer not null,
  home_team text not null,
  away_team text not null,
  kickoff_at text,
  venue text,
  toto_result text,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  primary key (round_no, match_no)
);

create index if not exists idx_toto_round_fixtures_round_no
on public.toto_round_fixtures(round_no);
