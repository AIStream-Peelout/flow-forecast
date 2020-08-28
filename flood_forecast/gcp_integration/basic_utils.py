from typing import Optional
from google.cloud import storage
import os
from oauthlib.service_account import ServiceAccountCredentials


def get_storage_client(
    service_key_path: Optional[str] = None,
) -> storage.Client:
    """
    Utility function to return a properly authenticated GCS
    storage client whether working in Colab, CircleCI, or other environment.
    """
    if service_key_path:
        # GOOGLE_APPLICATION_CREDENTIALS must be set
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_key_path
        return storage.Client()
    else:
        # credentials = os.environ["ENVIRONMENT_GCP"]
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            os.environ["ENVIRONMENT_GCP"]
        )
        return storage.Client(credentials=credentials)

        # try:
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["ENVIRONMENT_GCP"]
        # == "CircleCI":
        # creds = create_file_environ()
        # return storage.Client(
        #     credentials=creds, project=os.environ["GCP_PROJECT"]
        # )
        # return storage.Client()
        # elif: os.environ["ENVIRONMENT_GCP"] == "Colab":
        #     return storage.Client(project=os.environ["GCP_PROJECT"])


def upload_file(
    bucket_name: str, file_name: str, upload_name: str, client: storage.Client
):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(upload_name)


def create_file_environ():
    # TODO FIX
    from oauthlib.service_account import ServiceAccountCredentials

    credentials_dict = {
        "type": "service_account",
        "client_id": os.environ["BACKUP_CLIENT_ID"],
        "client_email": os.environ["BACKUP_CLIENT_EMAIL"],
        "private_key_id": os.environ["BACKUP_PRIVATE_KEY_ID"],
        "private_key": os.environ[
            "ENVIRONMENT_GCP"
        ],  # os.environ["BACKUP_PRIVATE_KEY"],
    }
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
        credentials_dict
    )
    return credentials


def download_file(
    bucket_name: str,
    source_blob_name: str,
    destination_file_name: str,
    service_key_path: Optional[str] = None,
):
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
