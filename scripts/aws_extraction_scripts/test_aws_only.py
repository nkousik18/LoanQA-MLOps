import os
import boto3
import json
from dotenv import load_dotenv

# 1. Load Environment (For AWS Keys only)
load_dotenv()


def test_aws_textract_logic():
    print("--- 🚀 Starting AWS-Only Test ---")

    # 2. Verify AWS Keys are loaded
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    bucket_name = os.getenv("AWS_SOURCE_BUCKET")
    region = os.getenv("AWS_REGION", "us-east-1")

    if not aws_key or not bucket_name:
        print("❌ CRITICAL: AWS_ACCESS_KEY_ID or AWS_SOURCE_BUCKET missing from .env")
        return

    try:
        # 3. Initialize AWS Clients
        s3 = boto3.client('s3', region_name=region)
        textract = boto3.client('textract', region_name=region)
        print("✅ AWS Clients Initialized")

        # 4. List ONE file from S3 to test connection
        print(f"🔎 Looking for files in S3 bucket: {bucket_name}...")
        response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=5)

        if 'Contents' not in response:
            print("⚠️ AWS Access OK, but Bucket is empty! Upload a PDF to S3 to test Textract.")
            return

        # Find a PDF
        pdf_key = None
        for obj in response['Contents']:
            if obj['Key'].lower().endswith('.pdf'):
                pdf_key = obj['Key']
                break

        if not pdf_key:
            print(
                f"⚠️ Found {len(response['Contents'])} files, but no PDFs. Found: {[o['Key'] for o in response['Contents']]}")
            return

        print(f"📄 Found PDF: {pdf_key}")

        # 5. Run Textract (The actual work)
        print("⚡ Sending to AWS Textract...")
        textract_response = textract.start_document_text_detection(
            DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': pdf_key}}
        )
        job_id = textract_response['JobId']
        print(f"✅ Textract Job Started! Job ID: {job_id}")

        # We won't wait for completion in this quick test,
        # just starting it proves permissions work.
        print("--- 🏁 Test Finished Successfully ---")
        print("Result: AWS Credentials & Textract API are working perfectly.")

    except Exception as e:
        print(f"\n❌ AWS TEST FAILED: {str(e)}")
        print("Check your AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and permissions.")


if __name__ == "__main__":
    test_aws_textract_logic()