# OpenManus-RL Supabase Integration

This document describes the Supabase integration for OpenManus-RL.

## Overview

The project is integrated with Supabase for:

- **PostgreSQL Database**: Storing training runs, rollouts, agent states, tool calls, and rewards
- **Storage**: Managing datasets, model checkpoints, logs, and evaluation results
- **Edge Functions**: API endpoints for rollout submission, state logging, and metrics

## Project Configuration

- **Project URL**: https://jdjhkmenfkmbaeaskkug.supabase.co
- **Project Ref**: jdjhkmenfkmbaeaskkug

## Database Schema

### Tables

1. **training_runs**: Training run metadata and configuration
2. **rollouts**: Individual episode executions
3. **agent_states**: Step-by-step agent observations and actions
4. **tool_calls**: Tool invocations during rollouts
5. **rewards**: Reward signals at each step
6. **model_checkpoints**: Model checkpoint metadata
7. **evaluation_results**: Evaluation metrics per checkpoint

### Migrations

All migrations are in `supabase/migrations/`:

- `20241109_initial_schema.sql`: Core database schema
- `20241109_storage_buckets.sql`: Storage bucket setup

## Storage Buckets

1. **datasets**: Training and evaluation datasets (50MB limit)
2. **model-checkpoints**: Model weights and parameters (1GB limit)
3. **logs**: Training and evaluation logs (10MB limit)
4. **evaluation-results**: Evaluation outputs (50MB limit)

## Edge Functions

### 1. submit-rollout

Create a new rollout episode.

**Endpoint**: `https://jdjhkmenfkmbaeaskkug.supabase.co/functions/v1/submit-rollout`

**Request**:

```json
{
  "training_run_id": "uuid",
  "episode_number": 1,
  "environment": "alfworld",
  "task_description": "Task description",
  "metadata": {}
}
```

**Response**:

```json
{
  "rollout_id": "uuid",
  "status": "created",
  "message": "Rollout created successfully"
}
```

### 2. log-agent-state

Log agent state for a rollout step.

**Endpoint**: `https://jdjhkmenfkmbaeaskkug.supabase.co/functions/v1/log-agent-state`

**Request**:

```json
{
  "rollout_id": "uuid",
  "step_number": 1,
  "observation": "Current observation",
  "action": "Action taken",
  "thought": "Agent reasoning",
  "memory_summary": "Memory state",
  "tool_calls": [
    {
      "tool_name": "google_search",
      "tool_input": { "query": "search term" },
      "tool_output": "Result",
      "success": true,
      "execution_time_ms": 150
    }
  ],
  "reward": {
    "value": 1.0,
    "type": "step",
    "details": {}
  }
}
```

### 3. complete-rollout

Mark a rollout as completed.

**Endpoint**: `https://jdjhkmenfkmbaeaskkug.supabase.co/functions/v1/complete-rollout`

**Request**:

```json
{
  "rollout_id": "uuid",
  "status": "success",
  "total_reward": 10.5,
  "step_count": 15
}
```

### 4. get-metrics

Get training metrics and statistics.

**Endpoint**: `https://jdjhkmenfkmbaeaskkug.supabase.co/functions/v1/get-metrics?training_run_id=uuid&environment=alfworld`

**Response**:

```json
{
  "total_rollouts": 100,
  "completed_rollouts": 95,
  "successful_rollouts": 80,
  "success_rate": 0.842,
  "average_reward": 8.5,
  "average_steps": 12.3,
  "rollouts": [...]
}
```

## Python Usage

### Setup

```python
from openmanus_rl.utils.supabase_client import get_supabase
from openmanus_rl.utils.supabase_db import TrainingRunManager, RolloutManager
from openmanus_rl.utils.supabase_storage import StorageManager
```

### Create Training Run

```python
run = TrainingRunManager.create_run(
    name="alfworld-gigpo-experiment",
    algorithm="gigpo",
    environment="alfworld",
    config={
        "learning_rate": 0.0001,
        "batch_size": 32
    }
)
```

### Create and Log Rollout

```python
# Create rollout
rollout = RolloutManager.create_rollout(
    training_run_id=run["id"],
    episode_number=1,
    environment="alfworld",
    task_description="Find the red ball"
)

# Log agent state
state_id = RolloutManager.log_agent_state(
    rollout_id=rollout["id"],
    step_number=1,
    observation="You are in a room",
    action="go north",
    thought="I should explore the room"
)

# Log tool call
RolloutManager.log_tool_call(
    agent_state_id=state_id,
    tool_name="google_search",
    tool_input={"query": "red ball location"},
    tool_output="Search results...",
    success=True,
    execution_time_ms=150
)

# Log reward
RolloutManager.log_reward(
    rollout_id=rollout["id"],
    step_number=1,
    reward=1.0,
    reward_type="step"
)

# Complete rollout
RolloutManager.complete_rollout(
    rollout_id=rollout["id"],
    status="success",
    total_reward=10.5,
    step_count=15
)
```

### Upload Files

```python
# Upload checkpoint
checkpoint_path = StorageManager.upload_checkpoint(
    training_run_id=run["id"],
    checkpoint_number=1,
    checkpoint_file="path/to/checkpoint.pt"
)

# Upload dataset
dataset_path = StorageManager.upload_dataset(
    environment="alfworld",
    dataset_name="train_data",
    dataset_file="data/alfworld/train.json"
)

# Upload log
log_path = StorageManager.upload_log(
    training_run_id=run["id"],
    log_type="training",
    log_file="logs/training.log"
)
```

## Environment Variables

Create a `.env.supabase` file:

```bash
SUPABASE_URL=https://jdjhkmenfkmbaeaskkug.supabase.co
SUPABASE_ANON_KEY=sb_publishable_IJFhatPZZcKJfB8G5QC9Tg_TqP4nTcX
SUPABASE_SERVICE_ROLE_KEY=sbp_ed71b8e9dd2c7d7205d626b99ad63a218934e67c
DATABASE_URL=postgresql://postgres:[YOUR_PASSWORD]@db.jdjhkmenfkmbaeaskkug.supabase.co:5432/postgres
SUPABASE_PROJECT_REF=jdjhkmenfkmbaeaskkug
```

## Deployment

### Manual Deployment

1. Install Supabase CLI:

   ```bash
   npm install -g supabase
   ```

2. Link project:

   ```bash
   supabase link --project-ref jdjhkmenfkmbaeaskkug
   ```

3. Push migrations:

   ```bash
   supabase db push
   ```

4. Deploy functions:
   ```bash
   supabase functions deploy submit-rollout
   supabase functions deploy log-agent-state
   supabase functions deploy complete-rollout
   supabase functions deploy get-metrics
   ```

### Automated Deployment

The project includes GitHub Actions workflows in `.github/workflows/supabase-deploy.yml` that automatically deploy migrations and functions on push to `main`.

**Required Secrets**:

- `SUPABASE_ACCESS_TOKEN`: Your Supabase access token
- `SUPABASE_SERVICE_ROLE_KEY`: Service role key for storage uploads

## Next Steps

1. Set your database password in `.env.supabase`
2. Run migrations: `supabase db push`
3. Deploy edge functions
4. Upload datasets: `python scripts/upload_datasets.py`
5. Update training scripts to use Supabase logging
