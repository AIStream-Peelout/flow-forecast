from typing import Optional
from google.cloud import storage
from google.oauth2.service_account import Credentials
import os


def get_storage_client(
    service_key_path: Optional[str] = None,
) -> storage.Client:
    """
    Utility function to return a properly authenticated GCS
    storage client whether working in Colab, CircleCI, Dataverse, or other environments.
    """
    if service_key_path is None:
        if os.environ["ENVIRONMENT_GCP"] == "Colab":
            return storage.Client(project=os.environ["GCP_PROJECT"])
        else:
            import ast
            cred_dict = ast.literal_eval(os.environ["ENVIRONMENT_GCP"])
            credentials = Credentials.from_service_account_info(cred_dict)
            return storage.Client(credentials=credentials, project=credentials.project_id)
    else:
        return storage.Client.from_service_account_json(service_key_path)


def upload_file(
    bucket_name: str, file_name: str, upload_name: str, client: storage.Client
):
    """A function to upload a file to a GCP bucket.

    :param bucket_name: The name of the bucket
    :type bucket_name: str
    :param file_name: The name of the file
    :type file_name: str
    :param upload_name: [description]
    :type upload_name: str
    :param client: [description]
    :type client: storage.Client
    """
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(upload_name)


def download_file(
    bucket_name: str,
    source_blob_name: str,
    destination_file_name: str,
    service_key_path: Optional[str] = None,
) -> None:
    """Download data file from GCS.

    Args:
        bucket_name ([str]): bucket name on GCS, eg. task_ts_data
        source_blob_name ([str]): storage object name
        destination_file_name ([str]): filepath to save to local
    """
    storage_client = get_storage_client(service_key_path)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )
