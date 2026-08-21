-- adaptive evidence layer: per-decision shadow records for governance.
-- run once in the supabase SQL editor. follows the repo's RLS pattern:
-- public read-only, writes only via the service key.

create table if not exists adaptive_evidence_log (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    log_date date not null,
    ticker text not null,
    action_proposed text not null check (action_proposed in ('BUY', 'SELL')),
    would_block boolean not null,
    armed_cited jsonb,
    cited jsonb,
    win_conf numeric,
    mode text not null check (mode in ('shadow', 'live'))
);

create index if not exists adaptive_log_date_idx
    on adaptive_evidence_log (log_date);
create index if not exists adaptive_log_ticker_idx
    on adaptive_evidence_log (ticker, log_date);

alter table adaptive_evidence_log enable row level security;

drop policy if exists "public read adaptive log" on adaptive_evidence_log;
create policy "public read adaptive log"
    on adaptive_evidence_log for select using (true);
