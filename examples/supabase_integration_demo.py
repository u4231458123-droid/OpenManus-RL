"""
OpenManus-RL Supabase Integration Example

This script demonstrates how to use the Supabase integration
for tracking training runs and rollouts.
"""

import os
from openmanus_rl.utils.supabase_db import TrainingRunManager, RolloutManager
from openmanus_rl.utils.supabase_storage import StorageManager


def main():
    """Example training run with Supabase logging."""

    print("🚀 Starting OpenManus-RL with Supabase Integration\n")

    # 1. Create a training run
    print("📝 Creating training run...")
    training_run = TrainingRunManager.create_run(
        name="demo-alfworld-gigpo",
        algorithm="gigpo",
        environment="alfworld",
        config={
            "learning_rate": 0.0001,
            "batch_size": 32,
            "num_epochs": 10,
            "gamma": 0.99
        }
    )
    print(f"✅ Training run created: {training_run['id']}\n")

    # 2. Create a rollout
    print("🎮 Starting rollout episode...")
    rollout = RolloutManager.create_rollout(
        training_run_id=training_run["id"],
        episode_number=1,
        environment="alfworld",
        task_description="Find the red ball in the living room",
        metadata={"difficulty": "easy", "scene": "living_room_1"}
    )
    print(f"✅ Rollout created: {rollout['id']}\n")

    # 3. Simulate some agent steps
    print("🤖 Simulating agent interactions...\n")

    for step in range(1, 4):
        print(f"Step {step}:")

        # Log agent state
        agent_state_id = RolloutManager.log_agent_state(
            rollout_id=rollout["id"],
            step_number=step,
            observation=f"You see a room with furniture. Step {step}.",
            action=f"go north" if step == 1 else f"look around",
            thought=f"I should explore the environment systematically.",
            memory_summary=f"Previous steps: {step-1}"
        )
        print(f"  📍 Agent state logged: {agent_state_id}")

        # Log a tool call
        RolloutManager.log_tool_call(
            agent_state_id=agent_state_id,
            tool_name="google_search" if step == 1 else "wikipedia_search",
            tool_input={"query": "how to find objects in rooms"},
            tool_output="Search results: ...",
            success=True,
            execution_time_ms=120 + step * 10
        )
        print(f"  🔧 Tool call logged")

        # Log reward
        reward_value = 0.5 if step < 3 else 5.0
        RolloutManager.log_reward(
            rollout_id=rollout["id"],
            step_number=step,
            reward=reward_value,
            reward_type="step" if step < 3 else "terminal",
            details={"reason": "exploration" if step < 3 else "task_complete"}
        )
        print(f"  🎁 Reward logged: {reward_value}\n")

    # 4. Complete the rollout
    print("🏁 Completing rollout...")
    RolloutManager.complete_rollout(
        rollout_id=rollout["id"],
        status="success",
        total_reward=6.0,
        step_count=3
    )
    print("✅ Rollout completed!\n")

    # 5. Update training run
    print("📊 Updating training run metrics...")
    TrainingRunManager.update_status(
        run_id=training_run["id"],
        status="completed",
        metrics={
            "success_rate": 1.0,
            "average_reward": 6.0,
            "total_episodes": 1
        }
    )
    print("✅ Training run updated!\n")

    # 6. Example: Upload a checkpoint (uncomment if you have a checkpoint file)
    # print("💾 Uploading model checkpoint...")
    # checkpoint_path = StorageManager.upload_checkpoint(
    #     training_run_id=training_run["id"],
    #     checkpoint_number=1,
    #     checkpoint_file="path/to/checkpoint.pt"
    # )
    # print(f"✅ Checkpoint uploaded to: {checkpoint_path}\n")

    print("=" * 50)
    print("🎉 Demo completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. View your data in Supabase Dashboard:")
    print("   https://supabase.com/dashboard/project/jdjhkmenfkmbaeaskkug")
    print("\n2. Query metrics via API:")
    print("   curl https://jdjhkmenfkmbaeaskkug.supabase.co/functions/v1/get-metrics")
    print("\n3. Integrate into your training loops!")


if __name__ == "__main__":
    # Make sure .env.supabase is loaded
    if not os.path.exists(".env.supabase"):
        print("❌ Error: .env.supabase not found!")
        print("Please create it with your Supabase credentials.")
        exit(1)

    main()
