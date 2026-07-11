-- key/value config table — paste into Supabase SQL Editor and run

create table if not exists config (
    key text primary key,
    value text not null
);

alter table config enable row level security;
create policy "public read config" on config
    for select using (true);
