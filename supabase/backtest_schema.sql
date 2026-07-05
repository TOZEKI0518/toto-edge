create table if not exists standings_history (
  id bigint generated always as identity primary key,
  round_no integer not null,
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

create index if not exists idx_standing_round
on standings_history(round_no);

create index if not exists idx_standing_team
on standings_history(team_name);

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