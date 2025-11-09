"""Script to upload datasets to Supabase Storage."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openmanus_rl.utils.supabase_storage import StorageManager


def upload_datasets():
    """Upload all datasets to Supabase storage."""

    data_dir = Path(__file__).parent.parent / "data"

    environments = {
        "gaia": ["val.json"],
        "alfworld": [],  # Add file patterns if available
        "webshop": [],
        "babyai": [],
        "maze": [],
        "movie": [],
        "sciworld": [],
        "sqlgym": [],
        "textcraft": [],
        "todo": [],
        "weather": [],
        "wordle": []
    }

    for env_name, files in environments.items():
        env_dir = data_dir / env_name

        if not env_dir.exists():
            print(f"Skipping {env_name}: directory not found")
            continue

        # If specific files listed, upload those
        if files:
            for filename in files:
                file_path = env_dir / filename
                if file_path.exists():
                    print(f"Uploading {env_name}/{filename}...")
                    try:
                        storage_path = StorageManager.upload_dataset(
                            environment=env_name,
                            dataset_name=Path(filename).stem,
                            dataset_file=str(file_path)
                        )
                        print(f"✓ Uploaded to: {storage_path}")
                    except Exception as e:
                        print(f"✗ Failed to upload {filename}: {e}")
                else:
                    print(f"File not found: {file_path}")
        else:
            # Upload all JSON files in directory
            json_files = list(env_dir.glob("*.json"))
            for file_path in json_files:
                print(f"Uploading {env_name}/{file_path.name}...")
                try:
                    storage_path = StorageManager.upload_dataset(
                        environment=env_name,
                        dataset_name=file_path.stem,
                        dataset_file=str(file_path)
                    )
                    print(f"✓ Uploaded to: {storage_path}")
                except Exception as e:
                    print(f"✗ Failed to upload {file_path.name}: {e}")


if __name__ == "__main__":
    print("Starting dataset upload to Supabase Storage...")
    upload_datasets()
    print("Dataset upload complete!")
