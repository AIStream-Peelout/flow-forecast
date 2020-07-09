from google.cloud import storage
import os


def get_storage_client() -> storage.Client:
    """
    Utility function to return a properly authenticated GCS
    storage client whether working in Colab, CircleCI, or other environment.
    """
    try:
        # GOOGLE_APPLICATION_CREDENTIALS must be set
        return storage.Client()
    except BaseException:
        if os.environ["ENVIRONMENT_GCP"] == "CircleCI":
            creds = create_file_environ()
            return storage.Client(credentials=creds, project=os.environ["GCP_PROJECT"])
        elif os.environ["ENVIRONMENT_GCP"] == "Colab":
            return storage.Client(project=os.environ["GCP_PROJECT"])


def upload_file(bucket_name: str, file_name: str, upload_name: str, client: storage.Client):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(upload_name)


def download_file(bucket_name, file_name, client, destination_file_name):
    bucket = client.get_bucket(bucket_name)
    # Create a blob object from the filepath
    blob = bucket.blob(file_name)
    # Download the file to a destination
    blob.download_to_filename(destination_file_name)
    print("File sucessfully downloaded")


def file_path_local_or_not(client, file_path: str):
    if "gs://" in file_path:
        print("Detected GCS URL will download file")
        file_path_split = file_path.split("gs://")[1].split("/")
        bucket_name = file_path_split[0]
        return download_file(bucket_name, "".join(file_path_split[1]), client, file_path_split[:-1])
    else:
        return file_path


def load_dataverse_file():
    pass
