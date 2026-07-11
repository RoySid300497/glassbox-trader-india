-- glassbox-trader memory schema
-- paste this into Supabase: SQL Editor -> New query -> Run

create table if not exists decisions (
    id bigint generated always as identity primary key,
    decided_at timestamptz not null default now(),
    ticker text not null,
    action text not null check (action in ('BUY','SELL','NO_TRADE')),
    cnn_direction text,
    cnn_confidence real,
    bull_case jsonb,
    bear_case jsonb,
    judge_votes jsonb,
    risk_gate_note text,
    outcome_return_1d real,
    outcome_return_5d real,
    outcome_label text,
    was_correct boolean,
    scored_at timestamptz
);

create table if not exists news_archive (
    id bigint generated always as identity primary key,
    ticker text not null,
    published_at timestamptz,
    source text,
    headline text not null,
    summary text,
    url text,
    sentiment real,
    fetched_at timestamptz not null default now(),
    unique (ticker, headline)
);

create table if not exists positions (
    ticker text primary key,
    qty real not null default 0,
    entry_price real,
    entry_date timestamptz,
    status text not null default 'OPEN' check (status in ('OPEN','CLOSED'))
);

create table if not exists market_context (
    date date primary key,
    summary_text text not null
);

create table if not exists lessons (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    lesson_text text not null,
    evidence jsonb,
    active boolean not null default true
);

create table if not exists theses (
    id bigint generated always as identity primary key,
    ticker text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    thesis_text text not null,
    direction text not null check (direction in ('LONG','SHORT')),
    evidence jsonb,
    confidence real not null default 0.5,
    status text not null default 'ACTIVE'
        check (status in ('ACTIVE','WEAKENING','CLOSED')),
    review_after date
);

create index if not exists idx_decisions_ticker on decisions (ticker, decided_at desc);
create index if not exists idx_news_ticker on news_archive (ticker, published_at desc);
create index if not exists idx_theses_ticker on theses (ticker) where status = 'ACTIVE';
