import pandas as pd
import requests
from datetime import datetime
from typing import Tuple, Dict
import pytz
from typing import Union


def make_usgs_data(start_date: datetime, end_date: datetime, site_number: str) -> pd.DataFrame:
    """Retrieves real-time streamflow data from the USGS National Water Information System (NWIS) for a specified site and date range.

    The data is saved locally as a raw response text file and then processed into a clean CSV.

    :param start_date: The start date for the data retrieval.
    :type start_date: datetime
    :param end_date: The end date for the data retrieval.
    :type end_date: datetime
    :param site_number: The 8-digit USGS site number (e.g., '02096750').
    :type site_number: str
    :return: A pandas DataFrame containing the raw flow data.
    :rtype: pd.DataFrame
    """
    base_url = "https://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&cb_00065&format=rdb&"
    full_url = base_url + "site_no=" + site_number + "&period=&begin_date=" + \
        start_date.strftime("%Y-%m-%d") + "&end_date=" + end_date.strftime("%Y-%m-%d")
    print("Getting request from USGS")
    print(full_url)
    r = requests.get(full_url)
    with open(site_number + ".txt", "w") as f:
        f.write(r.text)
    print("Request finished")
    response_data = process_response_text(site_number + ".txt")
    create_csv(response_data[0], response_data[1], site_number)
    return pd.read_csv(site_number + "_flow_data.csv")


def process_response_text(file_name: str) -> Tuple[str, Dict[str, str]]:
    """Processes the raw USGS response text file to extract metadata (like column names) and the data content.

    The data content is saved as a temporary TSV file for easy DataFrame loading.

    :param file_name: The path to the raw USGS response text file.
    :type file_name: str
    :return: A tuple containing the path to the temporary TSV data file and a dictionary mapping original USGS column names to simplified labels (e.g., 'cfs', 'height').
    :rtype: Tuple[str, Dict[str, str]]
    """
    extractive_params = {}
    with open(file_name, "r") as f:
        lines = f.readlines()
        i = 0
        params = False
        while "#" in lines[i]:
            # TODO figure out getting height and discharge code efficently
            the_split_line = lines[i].split()[1:]
            if params:
                print(the_split_line)
                if len(the_split_line) < 2:
                    params = False
                else:
                    extractive_params[the_split_line[0] + "_" + the_split_line[1]] = df_label(the_split_line[2])
            if len(the_split_line) > 2:
                if the_split_line[0] == "TS":
                    params = True
            i += 1
        tsv_file_path = file_name.split(".")[0] + "data.tsv"
        with open(tsv_file_path, "w") as t:
            # Write the data lines (after the metadata/comments) to a TSV file
            t.write("".join(lines[i:]))
        return tsv_file_path, extractive_params


def df_label(usgs_text: str) -> str:
    """Converts specific USGS parameter names into standardized, friendly column labels for the DataFrame.

    :param usgs_text: The original USGS text description of the parameter.
    :type usgs_text: str
    :return: The standardized column label ('cfs', 'height', or the original text if no standard mapping exists).
    :rtype: str
    """
    usgs_text = usgs_text.replace(",", "")
    if usgs_text == "Discharge":
        return "cfs"
    elif usgs_text == "Gage":
        return "height"
    else:
        return usgs_text


def create_csv(file_path: str, params_names: Dict[str, str], site_number: str):
    """Loads the intermediate TSV file, renames columns based on extracted parameters, and saves the final flow data CSV.

    :param file_path: Path to the intermediate TSV file containing the data.
    :type file_path: str
    :param params_names: Dictionary mapping original USGS parameter columns to new simplified names.
    :type params_names: Dict[str, str]
    :param site_number: The USGS site number used to name the output file.
    :type site_number: str
    """
    df = pd.read_csv(file_path, sep="\t")
    for key, value in params_names.items():
        df[value] = df[key]
    df.to_csv(site_number + "_flow_data.csv")


def get_timezone_map() -> Dict[str, str]:
    """Provides a mapping of USGS time zone abbreviations to IANA time zone strings."""
    timezone_map = {
        "EST": "America/New_York",
        "EDT": "America/New_York",
        "CST": "America/Chicago",
        "CDT": "America/Chicago",
        "MDT": "America/Denver",
        "MST": "America/Denver",
        "PST": "America/Los_Angeles",
        "PDT": "America/Los_Angeles"}
    return timezone_map


def process_intermediate_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    """Performs final cleanup and standardization on the flow data DataFrame, including time zone conversion and filtering.

    :param df: The raw DataFrame loaded from the site's flow data CSV.
    :type df: pd.DataFrame
    :return: A tuple containing the processed DataFrame (filtered to whole hours, in UTC), the maximum flow ('cfs') value, and the minimum flow ('cfs') value.
    :rtype: Tuple[pd.DataFrame, float, float]
    """
    # Remove garbage first row (often a header duplication)
    # TODO check if more rows are garbage
    df = df.iloc[1:]
    time_zone = df["tz_cd"].iloc[0]
    time_zone = get_timezone_map()[time_zone]
    old_timezone = pytz.timezone(time_zone)
    new_timezone = pytz.timezone("UTC")
    # Convert the local time column to UTC. This assumes timezones are consistent throughout the USGS stream (which should be true).
    df["datetime"] = df["datetime"].map(lambda x: old_timezone.localize(
        datetime.strptime(x, "%Y-%m-%d %H:%M")).astimezone(new_timezone))
    df["cfs"] = pd.to_numeric(df['cfs'], errors='coerce')
    max_flow = df["cfs"].max()
    min_flow = df["cfs"].min()
    count_nan = len(df["cfs"]) - df["cfs"].count()
    print(f"there are {count_nan} nan values")
    return df[df.datetime.dt.minute == 0], max_flow, min_flow
