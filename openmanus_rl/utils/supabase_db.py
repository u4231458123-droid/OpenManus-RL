"""Database models and operations for Supabase integration."""

from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID
from openmanus_rl.utils.supabase_client import get_supabase


class TrainingRunManager:
    """Manager for training run operations."""

    @staticmethod
    def create_run(
        name: str,
        algorithm: str,
        environment: str,
        config: Dict[str, Any]
    ) -> Dict:
        """Create a new training run."""
        supabase = get_supabase()

        data = {
            "name": name,
            "algorithm": algorithm,
            "environment": environment,
            "config": config,
            "status": "running"
        }

        response = supabase.table("training_runs").insert(data).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def update_status(run_id: str, status: str, metrics: Optional[Dict] = None):
        """Update training run status and metrics."""
        supabase = get_supabase()

        update_data = {"status": status}
        if metrics:
            update_data["metrics"] = metrics
        if status == "completed":
            update_data["completed_at"] = datetime.utcnow().isoformat()

        supabase.table("training_runs").update(update_data).eq("id", run_id).execute()


class RolloutManager:
    """Manager for rollout operations."""

    @staticmethod
    def create_rollout(
        training_run_id: Optional[str],
        episode_number: int,
        environment: str,
        task_description: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Create a new rollout."""
        supabase = get_supabase()

        data = {
            "training_run_id": training_run_id,
            "episode_number": episode_number,
            "environment": environment,
            "task_description": task_description,
            "status": "running",
            "metadata": metadata or {}
        }

        response = supabase.table("rollouts").insert(data).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def complete_rollout(
        rollout_id: str,
        status: str,
        total_reward: float,
        step_count: int
    ):
        """Complete a rollout."""
        supabase = get_supabase()

        update_data = {
            "status": status,
            "total_reward": total_reward,
            "step_count": step_count,
            "completed_at": datetime.utcnow().isoformat()
        }

        supabase.table("rollouts").update(update_data).eq("id", rollout_id).execute()

    @staticmethod
    def log_agent_state(
        rollout_id: str,
        step_number: int,
        observation: str,
        action: str,
        thought: Optional[str] = None,
        memory_summary: Optional[str] = None
    ) -> str:
        """Log an agent state."""
        supabase = get_supabase()

        data = {
            "rollout_id": rollout_id,
            "step_number": step_number,
            "observation": observation,
            "action": action,
            "thought": thought,
            "memory_summary": memory_summary
        }

        response = supabase.table("agent_states").insert(data).execute()
        return response.data[0]["id"] if response.data else None

    @staticmethod
    def log_tool_call(
        agent_state_id: str,
        tool_name: str,
        tool_input: Dict,
        tool_output: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None
    ):
        """Log a tool call."""
        supabase = get_supabase()

        data = {
            "agent_state_id": agent_state_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "success": success,
            "error_message": error_message,
            "execution_time_ms": execution_time_ms
        }

        supabase.table("tool_calls").insert(data).execute()

    @staticmethod
    def log_reward(
        rollout_id: str,
        step_number: int,
        reward: float,
        reward_type: str = "step",
        details: Optional[Dict] = None
    ):
        """Log a reward."""
        supabase = get_supabase()

        data = {
            "rollout_id": rollout_id,
            "step_number": step_number,
            "reward": reward,
            "reward_type": reward_type,
            "details": details or {}
        }

        supabase.table("rewards").insert(data).execute()


class ModelCheckpointManager:
    """Manager for model checkpoint operations."""

    @staticmethod
    def save_checkpoint(
        training_run_id: str,
        checkpoint_number: int,
        storage_path: str,
        metrics: Optional[Dict] = None
    ) -> Dict:
        """Save a model checkpoint reference."""
        supabase = get_supabase()

        data = {
            "training_run_id": training_run_id,
            "checkpoint_number": checkpoint_number,
            "storage_path": storage_path,
            "metrics": metrics or {}
        }

        response = supabase.table("model_checkpoints").insert(data).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def log_evaluation(
        model_checkpoint_id: str,
        environment: str,
        task_name: Optional[str],
        success: bool,
        score: Optional[float] = None,
        details: Optional[Dict] = None
    ):
        """Log an evaluation result."""
        supabase = get_supabase()

        data = {
            "model_checkpoint_id": model_checkpoint_id,
            "environment": environment,
            "task_name": task_name,
            "success": success,
            "score": score,
            "details": details or {}
        }

        supabase.table("evaluation_results").insert(data).execute()
