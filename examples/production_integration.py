"""
Production Integration Guide for OpenManus-RL with Vercel Dashboard

This script demonstrates how to integrate your training runs with the
live Vercel dashboard for real-time monitoring.
"""

import os
import time
from openmanus_rl.utils.supabase_db import TrainingRunManager, RolloutManager
from openmanus_rl.utils.supabase_storage import StorageManager


def create_training_run_with_monitoring(
    name: str,
    algorithm: str,
    environment: str,
    config: dict
):
    """
    Create a training run that will be visible on your Vercel dashboard.

    Args:
        name: Experiment name
        algorithm: Algorithm name (gigpo, ppo, etc.)
        environment: Environment name (alfworld, webshop, etc.)
        config: Training configuration dict

    Returns:
        Training run ID
    """
    print(f"\n🚀 Creating training run: {name}")
    print(f"   Algorithm: {algorithm}")
    print(f"   Environment: {environment}")

    # Create training run in Supabase
    run = TrainingRunManager.create_run(
        name=name,
        algorithm=algorithm,
        environment=environment,
        config=config
    )

    print(f"✅ Training run created: {run['id']}")
    print(f"📊 View on dashboard: [YOUR_VERCEL_URL]")

    return run['id']


def run_episode_with_logging(
    training_run_id: str,
    episode_number: int,
    environment: str,
    task_description: str = None
):
    """
    Run a single episode with full logging to dashboard.

    Args:
        training_run_id: Training run UUID
        episode_number: Episode number
        environment: Environment name
        task_description: Optional task description

    Returns:
        Rollout ID
    """
    print(f"\n🎮 Starting Episode #{episode_number}")

    # Create rollout
    rollout = RolloutManager.create_rollout(
        training_run_id=training_run_id,
        episode_number=episode_number,
        environment=environment,
        task_description=task_description,
        metadata={"version": "1.0"}
    )

    rollout_id = rollout['id']
    print(f"✅ Rollout created: {rollout_id}")

    # Simulate episode steps
    total_reward = 0

    for step in range(1, 6):  # 5 steps for demo
        print(f"   Step {step}/5...")

        # Log agent state
        state_id = RolloutManager.log_agent_state(
            rollout_id=rollout_id,
            step_number=step,
            observation=f"Observation at step {step}",
            action=f"Action {step}",
            thought=f"Reasoning for step {step}",
            memory_summary=f"Memory up to step {step}"
        )

        # Simulate tool call
        RolloutManager.log_tool_call(
            agent_state_id=state_id,
            tool_name="example_tool",
            tool_input={"query": f"query_{step}"},
            tool_output=f"result_{step}",
            success=True,
            execution_time_ms=100 + step * 10
        )

        # Log reward
        step_reward = 1.0 if step < 5 else 5.0
        RolloutManager.log_reward(
            rollout_id=rollout_id,
            step_number=step,
            reward=step_reward,
            reward_type="step" if step < 5 else "terminal"
        )

        total_reward += step_reward
        time.sleep(0.5)  # Simulate computation time

    # Complete rollout
    RolloutManager.complete_rollout(
        rollout_id=rollout_id,
        status="success",
        total_reward=total_reward,
        step_count=5
    )

    print(f"✅ Episode completed! Total reward: {total_reward}")
    print(f"📊 View on dashboard: [YOUR_VERCEL_URL]")

    return rollout_id


def integrate_with_existing_training_loop():
    """
    Example integration with your existing OpenManus-RL training loop.

    Add this to your training scripts to enable dashboard monitoring.
    """

    # Your existing training setup
    config = {
        "learning_rate": 0.0001,
        "batch_size": 32,
        "num_epochs": 10,
        "gamma": 0.99,
        "environment": "alfworld",
    }

    # Create training run
    run_id = create_training_run_with_monitoring(
        name="production-alfworld-gigpo",
        algorithm="gigpo",
        environment="alfworld",
        config=config
    )

    # Run episodes
    for episode in range(1, 4):  # 3 episodes for demo
        run_episode_with_logging(
            training_run_id=run_id,
            episode_number=episode,
            environment="alfworld",
            task_description=f"Task for episode {episode}"
        )

        time.sleep(1)  # Brief pause between episodes

    # Update final metrics
    TrainingRunManager.update_status(
        run_id=run_id,
        status="completed",
        metrics={
            "total_episodes": 3,
            "success_rate": 1.0,
            "average_reward": 9.0,
            "final_evaluation": "Excellent"
        }
    )

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"Training Run ID: {run_id}")
    print(f"Total Episodes: 3")
    print(f"Success Rate: 100%")
    print(f"Average Reward: 9.0")
    print("\n📊 View full results on your Vercel dashboard!")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("OpenManus-RL Production Integration Demo")
    print("="*60)
    print("\n⚠️  IMPORTANT: Update [YOUR_VERCEL_URL] with your actual Vercel URL")
    print("   Example: https://openmanus-rl-dashboard.vercel.app\n")

    # Run integration demo
    integrate_with_existing_training_loop()

    print("\n✅ Demo complete! Check your Vercel dashboard to see the results.")
    print("\n📚 Next steps:")
    print("   1. Replace [YOUR_VERCEL_URL] with your actual URL")
    print("   2. Integrate this pattern into your training scripts")
    print("   3. Monitor real-time progress on your dashboard")
    print("   4. Scale to full training runs")
