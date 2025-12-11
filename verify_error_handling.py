import unittest
from unittest.mock import patch, MagicMock
from botocore.exceptions import NoCredentialsError
from google.api_core.exceptions import NotFound

# Import your function
# Adjust the path if your structure is different
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'scripts', 'aws_extraction_scripts'))
from cloud_handler import transfer_s3_to_gcs


class TestCloudHandler(unittest.TestCase):

    @patch('cloud_handler.boto3.client')
    def test_aws_credentials_error(self, mock_boto):
        print("\n--- Testing AWS Credential Failure ---")
        # Simulate AWS saying "No Credentials"
        mock_boto.side_effect = NoCredentialsError()

        result = transfer_s3_to_gcs("test-bucket", "test.pdf", "gcs-bucket", "test.pdf")

        print(f"Result: {result}")
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['type'], 'AWS_ACCESS_ERROR')
        print("✅ AWS Error Handling: PASSED")

    @patch('cloud_handler.storage.Client')
    @patch('cloud_handler.boto3.client')
    def test_gcs_not_found_error(self, mock_boto, mock_gcs):
        print("\n--- Testing GCS Bucket Not Found ---")
        # AWS works fine
        mock_boto.return_value.get_object.return_value = {'Body': MagicMock(read=lambda: b'data')}

        # GCS fails with "Not Found"
        mock_gcs.side_effect = NotFound("Bucket not found")

        result = transfer_s3_to_gcs("test-bucket", "test.pdf", "gcs-bucket", "test.pdf")

        print(f"Result: {result}")
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['type'], 'GCS_ACCESS_ERROR')
        print("✅ GCS Error Handling: PASSED")


if __name__ == '__main__':
    unittest.main()