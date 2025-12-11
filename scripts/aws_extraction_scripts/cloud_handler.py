import os
import logging
import boto3
from google.cloud import storage
from botocore.exceptions import ClientError, NoCredentialsError
from google.api_core.exceptions import GoogleAPICallError, NotFound
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CloudTransfer")


def transfer_s3_to_gcs(s3_bucket, s3_key, gcs_bucket_name, gcs_blob_name):
    """
    Downloads a file from AWS S3 and uploads it to Google Cloud Storage.
    Returns a standardized dictionary for Frontend Error Handling.
    """
    try:
        # --- 1. AWS S3 Retrieval ---
        logger.info(f"Starting download: s3://{s3_bucket}/{s3_key}")

        # Initialize S3 Client (Auth is handled automatically by env vars)
        s3_client = boto3.client('s3')

        # Get the object
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        file_content = response['Body'].read()

        logger.info("Download from AWS successful.")

        # --- 2. GCS Upload ---
        logger.info(f"Starting upload: gs://{gcs_bucket_name}/{gcs_blob_name}")

        # Initialize GCS Client (Auth is handled automatically by env vars)
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)
        blob = bucket.blob(gcs_blob_name)

        # Upload contents
        blob.upload_from_string(file_content)

        logger.info("Upload to GCS successful.")

        return {
            "status": "success",
            "message": "File successfully moved from AWS to GCS",
            "s3_source": f"s3://{s3_bucket}/{s3_key}",
            "gcs_destination": f"gs://{gcs_bucket_name}/{gcs_blob_name}"
        }

    # --- ERROR HANDLING FOR FRONTEND ---

    except (NoCredentialsError, ClientError) as aws_e:
        logger.error(f"AWS Error: {str(aws_e)}")
        return {
            "status": "error",
            "type": "AWS_ACCESS_ERROR",
            "ui_message": "Could not access source file on AWS. Please check AWS keys and bucket permissions.",
            "technical_details": str(aws_e)
        }

    except (NotFound, GoogleAPICallError) as gcs_e:
        logger.error(f"GCS Error: {str(gcs_e)}")
        return {
            "status": "error",
            "type": "GCS_ACCESS_ERROR",
            "ui_message": "Failed to save file to Google Storage. Please check GCS Service Account permissions.",
            "technical_details": str(gcs_e)
        }

    except Exception as e:
        logger.critical(f"Critical System Error: {str(e)}")
        return {
            "status": "error",
            "type": "CRITICAL_SYSTEM_ERROR",
            "ui_message": "An unexpected system error occurred during file transfer.",
            "technical_details": str(e)
        }


# Optional: Simple test block to run this file directly
if __name__ == "__main__":
    # You can manually test by replacing these values with real ones temporarily
    # DO NOT COMMIT REAL VALUES TO GITHUB
    TEST_S3_BUCKET = os.getenv("AWS_SOURCE_BUCKET")
    TEST_S3_KEY = "test_document.pdf"  # Make sure this file exists in your S3
    TEST_GCS_BUCKET = os.getenv("GCS_DEST_BUCKET")
    TEST_GCS_BLOB = "test_document_copy.pdf"

    if TEST_S3_BUCKET and TEST_GCS_BUCKET:
        print("Running Test Transfer...")
        result = transfer_s3_to_gcs(TEST_S3_BUCKET, TEST_S3_KEY, TEST_GCS_BUCKET, TEST_GCS_BLOB)
        print(result)
    else:
        print("Skipping test: Environment variables for buckets not found.")