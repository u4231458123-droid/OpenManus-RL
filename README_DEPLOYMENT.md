# 🚀 NexifyAI OpenManus - Supabase Deployment

[![Supabase](https://img.shields.io/badge/Supabase-Integrated-green?style=flat&logo=supabase)](https://supabase.com)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Production-ready OpenManus-RL deployment with full Supabase integration for reinforcement learning experiments.

## 🎯 Features

- ✅ **PostgreSQL Database**: Complete schema for training runs, rollouts, agent states, rewards, and tool calls
- ✅ **Supabase Storage**: Organized buckets for datasets, model checkpoints, logs, and evaluation results
- ✅ **Edge Functions**: REST API endpoints for rollout submission, state logging, and metrics
- ✅ **Python Integration**: Ready-to-use managers for database and storage operations
- ✅ **GitHub Actions**: Automated deployment pipeline
- ✅ **Comprehensive Documentation**: Full setup and usage guides

## 🏗️ Architecture

```
OpenManus-RL + Supabase
├── PostgreSQL Database
│   ├── training_runs (experiments tracking)
│   ├── rollouts (episode execution)
│   ├── agent_states (step-by-step logs)
│   ├── tool_calls (tool usage tracking)
│   ├── rewards (reward signals)
│   └── model_checkpoints (checkpoint metadata)
│
├── Storage Buckets
│   ├── datasets (50MB limit)
│   ├── model-checkpoints (1GB limit)
│   ├── logs (10MB limit)
│   └── evaluation-results (50MB limit)
│
└── Edge Functions (REST APIs)
    ├── submit-rollout
    ├── log-agent-state
    ├── complete-rollout
    └── get-metrics
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Supabase CLI
npm install -g supabase

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configuration

The `.env.supabase` file is already configured with:

```bash
SUPABASE_URL=https://jdjhkmenfkmbaeaskkug.supabase.co
SUPABASE_PROJECT_REF=jdjhkmenfkmbaeaskkug
```

**Important**: Add your database password to `.env.supabase`

### 3. Deploy

#### Automated (Recommended)

```powershell
.\scripts\deploy_supabase.ps1
```

#### Manual

```bash
# Link project
supabase link --project-ref jdjhkmenfkmbaeaskkug

# Apply migrations
supabase db push

# Deploy functions
supabase functions deploy submit-rollout
supabase functions deploy log-agent-state
supabase functions deploy complete-rollout
supabase functions deploy get-metrics

# Upload datasets
python scripts/upload_datasets.py
```

## 💻 Usage Example

```python
from openmanus_rl.utils.supabase_db import TrainingRunManager, RolloutManager

# Create training run
run = TrainingRunManager.create_run(
    name="alfworld-gigpo-v1",
    algorithm="gigpo",
    environment="alfworld",
    config={"learning_rate": 0.0001}
)

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
    observation="You are in a living room",
    action="go north",
    thought="I should explore systematically"
)

# Complete rollout
RolloutManager.complete_rollout(
    rollout_id=rollout["id"],
    status="success",
    total_reward=10.5,
    step_count=15
)
```

See `examples/supabase_integration_demo.py` for a complete example.

## 📡 API Endpoints

All Edge Functions are available at:
```
https://jdjhkmenfkmbaeaskkug.supabase.co/functions/v1/
```

### Submit Rollout
```bash
POST /submit-rollout
{
  "episode_number": 1,
  "environment": "alfworld",
  "task_description": "Task description"
}
```

### Log Agent State
```bash
POST /log-agent-state
{
  "rollout_id": "uuid",
  "step_number": 1,
  "observation": "...",
  "action": "...",
  "tool_calls": [...]
}
```

### Complete Rollout
```bash
POST /complete-rollout
{
  "rollout_id": "uuid",
  "status": "success",
  "total_reward": 10.5,
  "step_count": 15
}
```

### Get Metrics
```bash
GET /get-metrics?environment=alfworld&training_run_id=uuid
```

## 📊 Database Schema

### Tables

- **training_runs**: Experiment metadata and configuration
- **rollouts**: Episode execution tracking
- **agent_states**: Detailed step-by-step logs
- **tool_calls**: Tool invocation history
- **rewards**: Reward signal tracking
- **model_checkpoints**: Checkpoint metadata
- **evaluation_results**: Evaluation metrics

All tables include Row Level Security (RLS) policies and indexes for optimal performance.

## 🗄️ Storage Buckets

| Bucket | Purpose | Size Limit |
|--------|---------|------------|
| `datasets` | Training/eval data | 50MB |
| `model-checkpoints` | Model weights | 1GB |
| `logs` | Training logs | 10MB |
| `evaluation-results` | Eval outputs | 50MB |

## 🔐 Security

- ✅ Row Level Security (RLS) enabled on all tables
- ✅ Service role for backend operations
- ✅ Anon key for frontend/API access
- ✅ Storage bucket policies configured

## 📚 Documentation

- [Supabase Integration Guide](docs/SUPABASE_INTEGRATION.md) - Full integration details
- [Quick Start Guide](SUPABASE_QUICKSTART.md) - Fast deployment guide
- [Original README](README.md) - OpenManus-RL documentation

## 🔧 Development

### Run Example
```bash
python examples/supabase_integration_demo.py
```

### Local Testing
```bash
# Start local Supabase
supabase start

# Run migrations
supabase db reset
```

## 🚢 CI/CD

GitHub Actions automatically deploys to Supabase on push to `main`:

1. Apply database migrations
2. Deploy Edge Functions
3. Upload datasets

**Required Secrets**:
- `SUPABASE_ACCESS_TOKEN`
- `SUPABASE_SERVICE_ROLE_KEY`

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Supabase Python Client
- **Database**: PostgreSQL 15 (Supabase)
- **Storage**: Supabase Storage (S3-compatible)
- **Edge Functions**: Deno runtime
- **CI/CD**: GitHub Actions
- **RL Framework**: OpenManus-RL (GIGPO, PPO)

## 📈 Monitoring

- **Supabase Dashboard**: https://supabase.com/dashboard/project/jdjhkmenfkmbaeaskkug
- **Metrics API**: `/functions/v1/get-metrics`
- **Logs**: Supabase Edge Function logs
- **Database**: PostgreSQL logs and analytics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `supabase start`
5. Submit a pull request

## 📝 License

Apache 2.0 - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL) - Original RL framework
- [Supabase](https://supabase.com) - Backend infrastructure
- Community contributors

## 📞 Support

- 📧 Issues: [GitHub Issues](https://github.com/u4231458123-droid/nexifyai-openmanus/issues)
- 📖 Docs: [Documentation](docs/)
- 💬 Discussions: [GitHub Discussions](https://github.com/u4231458123-droid/nexifyai-openmanus/discussions)

---

**Status**: ✅ Production Ready | 🚀 Deployed on Supabase

**Project URL**: https://jdjhkmenfkmbaeaskkug.supabase.co
