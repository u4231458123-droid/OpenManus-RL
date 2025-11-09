-- Enable necessary extensions
create extension if not exists "uuid-ossp";
create extension if not exists "pg_trgm";

-- Create training_runs table
create table if not exists public.training_runs (
    id uuid primary key default uuid_generate_v4(),
    name text not null,
    algorithm text not null, -- 'gigpo', 'ppo', etc.
    environment text not null, -- 'alfworld', 'webshop', 'gaia', etc.
    status text not null default 'running', -- 'running', 'completed', 'failed'
    config jsonb not null default '{}',
    metrics jsonb default '{}',
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
    completed_at timestamp with time zone
);

-- Create rollouts table
create table if not exists public.rollouts (
    id uuid primary key default uuid_generate_v4(),
    training_run_id uuid references public.training_runs(id) on delete cascade,
    episode_number integer not null,
    environment text not null,
    task_description text,
    status text not null default 'running', -- 'running', 'success', 'failed', 'timeout'
    total_reward numeric default 0,
    step_count integer default 0,
    started_at timestamp with time zone default timezone('utc'::text, now()) not null,
    completed_at timestamp with time zone,
    metadata jsonb default '{}'
);

-- Create agent_states table
create table if not exists public.agent_states (
    id uuid primary key default uuid_generate_v4(),
    rollout_id uuid references public.rollouts(id) on delete cascade not null,
    step_number integer not null,
    observation text,
    action text,
    thought text,
    memory_summary text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create tool_calls table
create table if not exists public.tool_calls (
    id uuid primary key default uuid_generate_v4(),
    agent_state_id uuid references public.agent_states(id) on delete cascade not null,
    tool_name text not null,
    tool_input jsonb not null,
    tool_output text,
    success boolean default true,
    error_message text,
    execution_time_ms integer,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create rewards table
create table if not exists public.rewards (
    id uuid primary key default uuid_generate_v4(),
    rollout_id uuid references public.rollouts(id) on delete cascade not null,
    step_number integer not null,
    reward numeric not null,
    reward_type text, -- 'step', 'terminal', 'shaped'
    details jsonb default '{}',
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create model_checkpoints table
create table if not exists public.model_checkpoints (
    id uuid primary key default uuid_generate_v4(),
    training_run_id uuid references public.training_runs(id) on delete cascade not null,
    checkpoint_number integer not null,
    storage_path text not null,
    metrics jsonb default '{}',
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create evaluation_results table
create table if not exists public.evaluation_results (
    id uuid primary key default uuid_generate_v4(),
    model_checkpoint_id uuid references public.model_checkpoints(id) on delete cascade,
    environment text not null,
    task_name text,
    success boolean not null,
    score numeric,
    details jsonb default '{}',
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create indexes for performance
create index if not exists idx_rollouts_training_run on public.rollouts(training_run_id);
create index if not exists idx_rollouts_status on public.rollouts(status);
create index if not exists idx_agent_states_rollout on public.agent_states(rollout_id);
create index if not exists idx_tool_calls_agent_state on public.tool_calls(agent_state_id);
create index if not exists idx_rewards_rollout on public.rewards(rollout_id);
create index if not exists idx_model_checkpoints_training_run on public.model_checkpoints(training_run_id);
create index if not exists idx_evaluation_results_checkpoint on public.evaluation_results(model_checkpoint_id);

-- Create updated_at trigger function
create or replace function public.handle_updated_at()
returns trigger as $$
begin
    new.updated_at = timezone('utc'::text, now());
    return new;
end;
$$ language plpgsql;

-- Add trigger to training_runs
create trigger set_updated_at
    before update on public.training_runs
    for each row
    execute function public.handle_updated_at();

-- Enable Row Level Security (RLS)
alter table public.training_runs enable row level security;
alter table public.rollouts enable row level security;
alter table public.agent_states enable row level security;
alter table public.tool_calls enable row level security;
alter table public.rewards enable row level security;
alter table public.model_checkpoints enable row level security;
alter table public.evaluation_results enable row level security;

-- Create policies (allow all for service role, restrict for anon)
create policy "Allow service role all access on training_runs"
    on public.training_runs for all
    using (true)
    with check (true);

create policy "Allow service role all access on rollouts"
    on public.rollouts for all
    using (true)
    with check (true);

create policy "Allow service role all access on agent_states"
    on public.agent_states for all
    using (true)
    with check (true);

create policy "Allow service role all access on tool_calls"
    on public.tool_calls for all
    using (true)
    with check (true);

create policy "Allow service role all access on rewards"
    on public.rewards for all
    using (true)
    with check (true);

create policy "Allow service role all access on model_checkpoints"
    on public.model_checkpoints for all
    using (true)
    with check (true);

create policy "Allow service role all access on evaluation_results"
    on public.evaluation_results for all
    using (true)
    with check (true);
