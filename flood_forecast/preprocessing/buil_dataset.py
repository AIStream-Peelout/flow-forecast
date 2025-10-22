import os
import re
from typing import Optional, Union
from pathlib import Path
from flood_forecast.preprocessing.closest_station import (
    get_weather_data,
    process_asos_data,
)
from flood_forecast.preprocessing.process_usgs import (
    make_usgs_data,
    process_intermediate_csv,
)
from flood_forecast.gcp_integration.basic_utils import (
    get_storage_client,
    upload_file,
    download_file,
)
from flood_forecast.preprocessing.eco_gage_set import eco_gage_set
import json
from datetime import datetime
import pytz
import pandas as pd
from typing import Tuple


def build_weather_csv(
    json_full_path: str,
    asos_base_url: str,
    base_url_2: str,
    econet_data: set,
    visited_gages_path: str,
    start: int = 0,
    end_index: int = 100,
):
    """Iterates through a directory of JSON files and retrieves/processes ASOS/ECO-NET weather data.

    :param json_full_path: The full path to the directory containing the JSON metadata files.
    :type json_full_path: str
    :param asos_base_url: The base URL for the ASOS data source.
    :type asos_base_url: str
    :param base_url_2: A secondary base URL for data processing (used in process_asos_data).
    :type base_url_2: str
    :param econet_data: A set of ECONET station IDs.
    :type econet_data: set
    :param visited_gages_path: Path to the JSON file tracking visited/completed gages.
    :type visited_gages_path: str
    :param start: The starting index for files in the directory list to process, defaults to 0.
    :type start: int
    :param end_index: The ending index (exclusive) for files in the directory list to process, defaults to 100.
    :type end_index: int
    """
    directory = os.fsencode(json_full_path)
    sorted_list = sorted(os.listdir(directory))
    for i in range(start, end_index):
        file = sorted_list[i]
        filename = os.fsdecode(file)
        get_weather_data(
            os.path.join(json_full_path, filename),
            econet_data,
            asos_base_url,
            visited_gages_path,
        )
        process_asos_data(
            os.path.join(json_full_path, filename),
            base_url_2,
            visited_gages_path,
        )


# todo fix this function so it does more than open files
# def make_usgs(meta_data_path: str, start, end_index: int):
#     meta_directory = os.fsencode(meta_data_path)
#     sorted_list = sorted(os.listdir(meta_directory))
#     for i in range(start, end_index):
#         with open(sorted_list[i]) as d:
#             data = json.loads(d)
#         # make_usgs_data(datetime(2014, 1, 1), datetime(2019,1,1), data["gage_id"])


def join_data(weather_csv: str, meta_json_file: str, flow_csv: str):
    """Placeholder function for joining different data sources.

    :param weather_csv: Path to the weather data CSV file.
    :type weather_csv: str
    :param meta_json_file: Path to the metadata JSON file.
    :type meta_json_file: str
    :param flow_csv: Path to the streamflow data CSV file.
    :type flow_csv: str
    """
    pass


def create_visited():
    """Creates a fresh `visited_gages.json` file with empty station tracking objects.
    """
    visited_gages = {"stations_visited": {}, "saved_complete": {}}
    with open("visited_gages.json", "w+") as f:
        json.dump(visited_gages, f)


def get_eco_netset(directory_path: str) -> set:
    """Econet data was supplied to us by the NC State climate office.

    They gave us a directory of CSV files in following format `LastName_First_station_id_Hourly.txt` This code simply
    constructs a set of stations based on what is in the folder.

    :param directory_path: The path to the directory containing ECONET CSV files.
    :type directory_path: str
    :return: A set containing the unique ECONET station IDs extracted from the filenames.
    :rtype: set
    """
    directory = os.fsencode(directory_path)
    print(sorted(os.listdir(directory)))
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        try:
            eco_gage_set.add(filename.split("c_")[1].split("_H")[0])
        except BaseException:
            print(filename)
    return eco_gage_set


def combine_data(flow_df: pd.DataFrame, precip_df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """Combines streamflow and precipitation dataframes based on a shared timestamp.

    :param flow_df: A pandas DataFrame containing streamflow data with a 'datetime' column.
    :type flow_df: pd.DataFrame
    :param precip_df: A pandas DataFrame containing precipitation data with an 'hour_updated' column.
    :type precip_df: pd.DataFrame
    :return: A tuple containing the joined DataFrame, the number of NaN values in the 'cfs' column (nan_flow), and the number of NaN values in the 'p01m' column (nan_precip).
    :rtype: Tuple[pd.DataFrame, int, int]
    """
    tz = pytz.timezone("UTC")
    precip_df["hour_updated"] = precip_df["hour_updated"].map(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    )
    precip_df["hour_updated"] = precip_df["hour_updated"].map(
        lambda x: tz.localize(x)
    )
    joined_df = precip_df.merge(
        flow_df, left_on="hour_updated", right_on="datetime", how="outer"
    )[4:-4]
    nan_precip = sum(pd.isnull(joined_df["p01m"]))
    nan_flow = sum(pd.isnull(joined_df["cfs"]))
    return joined_df, nan_flow, nan_precip


def create_usgs(meta_data_dir: str, precip_path: str, start: int, end: int):
    """Retrieves USGS streamflow data, combines it with local precipitation data, updates metadata, and uploads results to GCS.

    :param meta_data_dir: Directory containing USGS station metadata JSON files.
    :type meta_data_dir: str
    :param precip_path: Directory containing local precipitation CSV files.
    :type precip_path: str
    :param start: The starting index for files in the directory list to process.
    :type start: int
    :param end: The ending index (exclusive) for files in the directory list to process.
    :type end: int
    """
    gage_list = sorted(os.listdir(meta_data_dir))
    exceptions = {}
    client = get_storage_client()
    for i in range(start, end):
        try:
            file_name = gage_list[i]
            gage_id = file_name.split("stations")[0]
            with open(os.path.join(meta_data_dir, file_name)) as f:
                print(os.path.join(meta_data_dir, file_name))
                data = json.load(f)
            if len(gage_id) == 7:
                gage_id = "0" + gage_id
                raw_df = make_usgs_data(
                    datetime(2014, 1, 1), datetime(2019, 1, 1), gage_id
                )
            else:
                raw_df = make_usgs_data(
                    datetime(2014, 1, 1), datetime(2019, 1, 1), gage_id
                )
            df, max_flow, min_flow = process_intermediate_csv(raw_df)
            data["time_zone_code"] = df["tz_cd"].iloc[0]
            data["max_flow"] = max_flow
            data["min_flow"] = min_flow
            precip_df = pd.read_csv(
                os.path.join(
                    precip_path, data["stations"][0]["station_id"] + ".csv"
                )
            )
            fixed_df, nan_flow, nan_precip = combine_data(df, precip_df)
            data["nan_flow"] = nan_flow
            data["nan_precip"] = nan_precip
            joined_name = (
                str(gage_id) + data["stations"][0]["station_id"] + "_flow.csv"
            )
            joined_upload = "joined/" + joined_name
            meta_path = os.path.join(meta_data_dir, file_name)
            data["files"] = [joined_name]
            fixed_df.to_csv(joined_name)
            with open(meta_path, "w") as f:
                json.dump(data, f)
            upload_file("predict_cfs", "meta2/" + file_name, meta_path, client)
            upload_file("predict_cfs", joined_upload, joined_name, client)
        except Exception as e:
            exceptions[str(gage_id)] = str(e)
            with open("exceptions.json", "w+") as a:
                json.dump(exceptions, a)
            print("exception")
            upload_file(
                "predict_cfs",
                "meta2/" + "exceptions.json",
                "exceptions.json",
                client,
            )


def get_data(file_path: str, gcp_service_key: Optional[str] = None) -> Union[str, pd.DataFrame]:
    """Downloads data from a Google Cloud Storage (GCS) path or reads a local file.

    Extracts bucket name and storage object name from file_path if it's a GCS path.

    :param file_path: The path to the data file. Can be a local path or a GCS URI (e.g., "gs://bucket/object.csv").
    :type file_path: str
    :param gcp_service_key: Optional path to the GCP service account key JSON file for authentication.
    :type gcp_service_key: Optional[str]
    :return: Either a pandas DataFrame if the file is a CSV, or the local file path string if it's a non-CSV file or the input was already a DataFrame.
    :rtype: Union[str, pd.DataFrame]
    """
    if isinstance(file_path, pd.DataFrame):
        return file_path
    if file_path.startswith("gs://"):
        # download data from gcs to local
        print(file_path)
        regex = r"(?<=gs:\/\/)[a-zA-Z0-9\-\_]*(?=\/)"
        bucket_name = re.search(regex, file_path).group()
        object_name = re.search(rf"(?<={bucket_name}\/).*", file_path).group()
        local_temp_filepath = Path("data") / bucket_name / object_name
        if not local_temp_filepath.parent.exists():
            local_temp_filepath.parent.mkdir(parents=True, exist_ok=True)

        download_file(
            bucket_name=bucket_name,
            source_blob_name=object_name,
            destination_file_name=local_temp_filepath,
            service_key_path=gcp_service_key,
        )
        if str(local_temp_filepath)[-3:] != "csv":
            return local_temp_filepath
        return pd.read_csv(str(local_temp_filepath))
    elif str(file_path)[-3:] != "csv":
        return file_path
    return pd.read_csv(file_path)