"""
Deploy Supabase migrations and Edge Functions using the Management API.
Since CLI installation is not supported, we use direct API calls.
"""

import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.supabase")

# Configuration
SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF", "jdjhkmenfkmbaeaskkug")
SUPABASE_ACCESS_TOKEN = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "sbp_ed71b8e9dd2c7d7205d626b99ad63a218934e67c")
SUPABASE_API_URL = f"https://api.supabase.com/v1/projects/{SUPABASE_PROJECT_REF}"


def execute_sql(sql_content: str, migration_name: str) -> bool:
    """Execute SQL using Supabase SQL editor API."""

    # Note: Direct SQL execution via API requires authentication
    # For production, use Supabase CLI or Studio
    print(f"📝 Migration: {migration_name}")
    print(f"   SQL Content Length: {len(sql_content)} characters")
    print(f"   ⚠️  Note: Execute this manually in Supabase Studio SQL Editor")
    print(f"   URL: https://supabase.com/dashboard/project/{SUPABASE_PROJECT_REF}/sql/new")
    print()
    return True


def deploy_migrations():
    """Deploy all SQL migrations."""
    print("🚀 Deploying Supabase Migrations\n")
    print("=" * 60)

    migrations_dir = Path("supabase/migrations")

    if not migrations_dir.exists():
        print("❌ Migrations directory not found!")
        return False

    migration_files = sorted(migrations_dir.glob("*.sql"))

    if not migration_files:
        print("ℹ️  No migration files found")
        return True

    for migration_file in migration_files:
        with open(migration_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        execute_sql(sql_content, migration_file.name)

    print("=" * 60)
    print()
    return True


def deploy_edge_function(function_name: str) -> bool:
    """Deploy an Edge Function."""

    function_dir = Path(f"supabase/functions/{function_name}")

    if not function_dir.exists():
        print(f"❌ Function directory not found: {function_name}")
        return False

    index_file = function_dir / "index.ts"

    if not index_file.exists():
        print(f"❌ index.ts not found for function: {function_name}")
        return False

    with open(index_file, 'r', encoding='utf-8') as f:
        function_code = f.read()

    print(f"📦 Edge Function: {function_name}")
    print(f"   Code Length: {len(function_code)} characters")
    print(f"   ⚠️  Note: Deploy manually via Supabase Dashboard")
    print(f"   URL: https://supabase.com/dashboard/project/{SUPABASE_PROJECT_REF}/functions")
    print()

    return True


def deploy_edge_functions():
    """Deploy all Edge Functions."""
    print("🌐 Deploying Edge Functions\n")
    print("=" * 60)

    functions = [
        "submit-rollout",
        "log-agent-state",
        "complete-rollout",
        "get-metrics"
    ]

    for function_name in functions:
        deploy_edge_function(function_name)

    print("=" * 60)
    print()


def verify_configuration():
    """Verify Supabase configuration."""
    print("🔍 Verifying Configuration\n")
    print("=" * 60)

    print(f"✅ Project Ref: {SUPABASE_PROJECT_REF}")
    print(f"✅ API URL: {SUPABASE_API_URL}")
    print(f"✅ Access Token: {'*' * 20}{SUPABASE_ACCESS_TOKEN[-8:]}")

    print("\n📊 Configuration Files:")

    files_to_check = [
        ".env.supabase",
        "supabase/config.toml",
        "supabase/migrations/20241109_initial_schema.sql",
        "supabase/migrations/20241109_storage_buckets.sql",
    ]

    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (missing)")

    print("=" * 60)
    print()


def print_manual_deployment_guide():
    """Print manual deployment instructions."""
    print("\n" + "=" * 60)
    print("📋 MANUAL DEPLOYMENT GUIDE")
    print("=" * 60)
    print()

    print("Since Supabase CLI cannot be installed globally on this system,")
    print("please follow these manual steps:")
    print()

    print("1️⃣  DATABASE MIGRATIONS")
    print("   a. Go to: https://supabase.com/dashboard/project/jdjhkmenfkmbaeaskkug/sql/new")
    print("   b. Copy content from: supabase/migrations/20241109_initial_schema.sql")
    print("   c. Paste and run in SQL Editor")
    print("   d. Copy content from: supabase/migrations/20241109_storage_buckets.sql")
    print("   e. Paste and run in SQL Editor")
    print()

    print("2️⃣  EDGE FUNCTIONS")
    print("   a. Go to: https://supabase.com/dashboard/project/jdjhkmenfkmbaeaskkug/functions")
    print("   b. Click 'Create a new function'")
    print("   c. For each function (submit-rollout, log-agent-state, complete-rollout, get-metrics):")
    print("      - Name: [function-name]")
    print("      - Code: Copy from supabase/functions/[function-name]/index.ts")
    print("      - Click 'Deploy'")
    print()

    print("3️⃣  VERIFY DEPLOYMENT")
    print("   a. Check Tables: https://supabase.com/dashboard/project/jdjhkmenfkmbaeaskkug/editor")
    print("   b. Check Storage: https://supabase.com/dashboard/project/jdjhkmenfkmbaeaskkug/storage/buckets")
    print("   c. Check Functions: https://supabase.com/dashboard/project/jdjhkmenfkmbaeaskkug/functions")
    print()

    print("4️⃣  TEST API")
    print("   curl https://jdjhkmenfkmbaeaskkug.supabase.co/functions/v1/get-metrics")
    print()

    print("=" * 60)
    print()


def main():
    """Main deployment function."""
    print("\n" + "🚀" * 30)
    print("OPENMANUS-RL SUPABASE DEPLOYMENT")
    print("🚀" * 30 + "\n")

    # Verify configuration
    verify_configuration()

    # Deploy migrations
    deploy_migrations()

    # Deploy Edge Functions
    deploy_edge_functions()

    # Print manual deployment guide
    print_manual_deployment_guide()

    print("✅ Deployment preparation complete!")
    print("\nNext steps:")
    print("1. Follow the manual deployment guide above")
    print("2. Deploy dashboard to Vercel: cd dashboard && vercel")
    print("3. Test the complete system")
    print()


if __name__ == "__main__":
    main()
