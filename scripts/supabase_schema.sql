-- Supabase schema for the live signal log mirror.
-- Run once in the Supabase SQL editor (Project → SQL) before the first sync.
--
-- Security model:
--   * The GitHub Action writes with the SERVICE ROLE key (bypasses RLS).
--   * RLS is enabled so the public anon key can only SELECT — safe to use
--     from a read-only dashboard.

create table if not exists live_signals (
    ticker      text        not null,
    date        text        not null,
    direction   integer     not null,
    label       text        not null,
    confidence  double precision,
    p_up        double precision,
    p_flat      double precision,
    p_down      double precision,
    kelly_frac  double precision,
    atr         double precision,
    stop_loss   double precision,
    close       double precision,
    run_ts      text        not null,
    primary key (ticker, run_ts)
);

create table if not exists live_orders (
    order_id    text,
    ticker      text        not null,
    side        text        not null,
    qty         double precision,
    order_type  text,
    status      text,
    stop_price  double precision,
    take_profit double precision,
    error       text,
    run_ts      text        not null,
    primary key (ticker, run_ts)
);

create table if not exists live_equity (
    equity        double precision not null,
    buying_power  double precision,
    run_ts        text             not null,
    primary key (run_ts)
);

-- Read-only public access; writes only via service role.
alter table live_signals enable row level security;
alter table live_orders  enable row level security;
alter table live_equity  enable row level security;

create policy "public read signals" on live_signals for select using (true);
create policy "public read orders"  on live_orders  for select using (true);
create policy "public read equity"  on live_equity  for select using (true);
