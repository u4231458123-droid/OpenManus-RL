"""Storage utilities for Supabase integration."""

import os
from typing import BinaryIO, Optional
from pathlib import Path
from openmanus_rl.utils.supabase_client import get_supabase


class StorageManager:
    """Manager for Supabase storage operations."""

    BUCKETS = {
        "datasets": "datasets",
        "checkpoints": "model-checkpoints",
        "logs": "logs",
        "evaluations": "evaluation-results"
    }

    @staticmethod
    def upload_file(
        bucket: str,
        file_path: str,
        destination_path: str,
        content_type: Optional[str] = None
    ) -> str:
        """Upload a file to Supabase storage.

        Args:
            bucket: Bucket name (use BUCKETS keys)
            file_path: Local file path to upload
            destination_path: Destination path in bucket
            content_type: MIME type of the file

        Returns:
            Public URL of the uploaded file
        """
        supabase = get_supabase()
        bucket_name = StorageManager.BUCKETS.get(bucket, bucket)

        with open(file_path, 'rb') as f:
            file_data = f.read()

        options = {}
        if content_type:
            options['content-type'] = content_type

        supabase.storage.from_(bucket_name).upload(
            destination_path,
            file_data,
            file_options=options
        )

        # Get public URL
        public_url = supabase.storage.from_(bucket_name).get_public_url(destination_path)
        return public_url

    @staticmethod
    def upload_checkpoint(
        training_run_id: str,
        checkpoint_number: int,
        checkpoint_file: str
    ) -> str:
        """Upload a model checkpoint.

        Args:
            training_run_id: Training run UUID
            checkpoint_number: Checkpoint iteration number
            checkpoint_file: Path to checkpoint file

        Returns:
            Storage path in bucket
        """
        destination = f"{training_run_id}/checkpoint_{checkpoint_number}.pt"

        StorageManager.upload_file(
            "checkpoints",
            checkpoint_file,
            destination,
            content_type="application/octet-stream"
        )

        return destination

    @staticmethod
    def upload_dataset(
        environment: str,
        dataset_name: str,
        dataset_file: str
    ) -> str:
        """Upload a dataset file.

        Args:
            environment: Environment name (alfworld, webshop, etc.)
            dataset_name: Dataset identifier
            dataset_file: Path to dataset file

        Returns:
            Storage path in bucket
        """
        file_ext = Path(dataset_file).suffix
        destination = f"{environment}/{dataset_name}{file_ext}"

        content_type = "application/json" if file_ext == ".json" else None

        StorageManager.upload_file(
            "datasets",
            dataset_file,
            destination,
            content_type=content_type
        )

        return destination

    @staticmethod
    def upload_log(
        training_run_id: str,
        log_type: str,
        log_file: str
    ) -> str:
        """Upload a log file.

        Args:
            training_run_id: Training run UUID
            log_type: Type of log (training, evaluation, etc.)
            log_file: Path to log file

        Returns:
            Storage path in bucket
        """
        file_ext = Path(log_file).suffix
        destination = f"{training_run_id}/{log_type}{file_ext}"

        StorageManager.upload_file(
            "logs",
            log_file,
            destination,
            content_type="text/plain"
        )

        return destination

    @staticmethod
    def upload_evaluation_result(
        model_checkpoint_id: str,
        environment: str,
        result_file: str
    ) -> str:
        """Upload an evaluation result file.

        Args:
            model_checkpoint_id: Model checkpoint UUID
            environment: Environment name
            result_file: Path to result file

        Returns:
            Storage path in bucket
        """
        file_ext = Path(result_file).suffix
        destination = f"{model_checkpoint_id}/{environment}_results{file_ext}"

        content_type = "application/json" if file_ext == ".json" else None

        StorageManager.upload_file(
            "evaluations",
            result_file,
            destination,
            content_type=content_type
        )

        return destination

    @staticmethod
    def download_file(
        bucket: str,
        source_path: str,
        destination_path: str
    ):
        """Download a file from Supabase storage.

        Args:
            bucket: Bucket name (use BUCKETS keys)
            source_path: Source path in bucket
            destination_path: Local destination path
        """
        supabase = get_supabase()
        bucket_name = StorageManager.BUCKETS.get(bucket, bucket)

        response = supabase.storage.from_(bucket_name).download(source_path)

        with open(destination_path, 'wb') as f:
            f.write(response)

    @staticmethod
    def list_files(bucket: str, path: str = "") -> list:
        """List files in a bucket path.

        Args:
            bucket: Bucket name (use BUCKETS keys)
            path: Path prefix to list

        Returns:
            List of file objects
        """
        supabase = get_supabase()
        bucket_name = StorageManager.BUCKETS.get(bucket, bucket)

        files = supabase.storage.from_(bucket_name).list(path)
        return files
