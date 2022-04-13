import pandas as pd
import requests
from datetime import datetime
from typing import Tuple, Dict
import pytz
# url format


def make_usgs_data(start_date: datetime, end_date: datetime, site_number: str) -> pd.DataFrame:
    """"""
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


def process_response_text(file_name: str) -> Tuple[str, Dict]:
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
        with open(file_name.split(".")[0] + "data.tsv", "w") as t:
            t.write("".join(lines[i:]))
        return file_name.split(".")[0] + "data.tsv", extractive_params


def df_label(usgs_text: str) -> str:
    usgs_text = usgs_text.replace(",", "")
    if usgs_text == "Discharge":
        return "cfs"
    elif usgs_text == "Gage":
        return "height"
    else:
        return usgs_text


def create_csv(file_path: str, params_names: dict, site_number: str):
    """
    Function that creates the final version of the CSV files
    """
    df = pd.read_csv(file_path, sep="\t")
    for key, value in params_names.items():
        df[value] = df[key]
    df.to_csv(site_number + "_flow_data.csv")


def get_timezone_map():
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


def process_intermediate_csv(df: pd.DataFrame) -> (pd.DataFrame, int, int, int):
    # Remove garbage first row
    # TODO check if more rows are garbage
    df = df.iloc[1:]
    time_zone = df["tz_cd"].iloc[0]
    time_zone = get_timezone_map()[time_zone]
    old_timezone = pytz.timezone(time_zone)
    new_timezone = pytz.timezone("UTC")
    # This assumes timezones are consistent throughout the USGS stream (this should be true)
    df["datetime"] = df["datetime"].map(lambda x: old_timezone.localize(
        datetime.strptime(x, "%Y-%m-%d %H:%M")).astimezone(new_timezone))
    df["cfs"] = pd.to_numeric(df['cfs'], errors='coerce')
    max_flow = df["cfs"].max()
    min_flow = df["cfs"].min()
    count_nan = len(df["cfs"]) - df["cfs"].count()
    print(f"there are {count_nan} nan values")
    return df[df.datetime.dt.minute == 0], max_flow, min_flow
