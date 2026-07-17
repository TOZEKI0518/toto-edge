create table if not exists public.toto_round_runs (
    round_id bigint primary key,
    decision text not null,
    reason text not null,
    investment_yen integer not null default 0,
    ticket_count integer not null default 0,
    available_budget_yen integer not null default 0,
    normal_budget_yen integer not null default 0,
    rollover_before_yen integer not null default 0,
    rollover_after_yen integer not null default 0,
    estimated_portfolio_value_index double precision,
    estimated_expected_return_yen double precision,
    estimated_expected_profit_yen double precision,
    model_coverage_probability double precision not null default 0,
    market_coverage_probability double precision not null default 0,
    high_confidence_matches integer not null default 0,
    medium_confidence_matches integer not null default 0,
    low_confidence_matches integer not null default 0,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.toto_round_match_predictions (
    round_id bigint not null,
    toto_match_no integer not null check (toto_match_no between 1 and 13),
    match_card_id bigint not null,
    match_date date,
    home_team text not null,
    away_team text not null,

    prediction text not null check (prediction in ('A', 'D', 'H')),
    prob_away double precision not null,
    prob_draw double precision not null,
    prob_home double precision not null,

    rf_prediction text,
    rf_prob_away double precision,
    rf_prob_draw double precision,
    rf_prob_home double precision,

    lgbm_prediction text,
    lgbm_prob_away double precision,
    lgbm_prob_draw double precision,
    lgbm_prob_home double precision,

    market_prob_away double precision not null,
    market_prob_draw double precision not null,
    market_prob_home double precision not null,

    best_edge double precision,
    best_value_ratio double precision,
    best_value_pick text,
    value_score double precision,
    roi_priority_score double precision,

    models_agree boolean,
    ai_market_agree boolean,

    confidence_score double precision,
    confidence_level text,
    coverage_recommendation text,
    recommended_combination text,
    primary_pick text,
    secondary_pick text,
    model_disagreement_js double precision,

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    primary key (round_id, toto_match_no)
);

create table if not exists public.toto_round_tickets (
    round_id bigint not null,
    ticket_number integer not null,
    ticket_cost_yen integer not null default 100,
    picks text not null,
    model_probability double precision,
    market_probability double precision,
    conservative_value_index double precision,
    search_score double precision,

    match_01 text,
    match_02 text,
    match_03 text,
    match_04 text,
    match_05 text,
    match_06 text,
    match_07 text,
    match_08 text,
    match_09 text,
    match_10 text,
    match_11 text,
    match_12 text,
    match_13 text,

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    primary key (round_id, ticket_number)
);

create index if not exists idx_toto_round_runs_updated_at
    on public.toto_round_runs(updated_at desc);

create index if not exists idx_toto_round_match_predictions_round
    on public.toto_round_match_predictions(round_id);

create index if not exists idx_toto_round_match_predictions_value
    on public.toto_round_match_predictions(
        round_id,
        roi_priority_score desc
    );

create index if not exists idx_toto_round_tickets_round
    on public.toto_round_tickets(round_id);

alter table public.toto_round_runs enable row level security;
alter table public.toto_round_match_predictions enable row level security;
alter table public.toto_round_tickets enable row level security;

drop policy if exists "Public read toto round runs"
    on public.toto_round_runs;
create policy "Public read toto round runs"
    on public.toto_round_runs
    for select
    using (true);

drop policy if exists "Public read toto match predictions"
    on public.toto_round_match_predictions;
create policy "Public read toto match predictions"
    on public.toto_round_match_predictions
    for select
    using (true);

drop policy if exists "Public read toto round tickets"
    on public.toto_round_tickets;
create policy "Public read toto round tickets"
    on public.toto_round_tickets
    for select
    using (true);
