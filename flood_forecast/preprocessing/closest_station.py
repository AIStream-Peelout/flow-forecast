from math import radians, cos, sin, asin, sqrt
import pandas as pd
import os
import json
from typing import Set, Dict, Tuple
import requests
from datetime import datetime, timedelta


def get_closest_gage(
        gage_df: pd.DataFrame,
        station_df: pd.DataFrame,
        path_dir: str,
        start_row: int,
        end_row: int):
    """Calculates the distance between river gages and weather stations (ASOS/ECO-NET) using the Haversine formula, and stores the results in JSON files.

    :param gage_df: DataFrame containing river gage information (including 'id', 'latitude', 'logitude').
    :type gage_df: pd.DataFrame
    :param station_df: DataFrame containing weather station information (including 'lon', 'lat', 'stid').
    :type station_df: pd.DataFrame
    :param path_dir: The directory path where the resulting JSON metadata files should be saved.
    :type path_dir: str
    :param start_row: The starting row index (inclusive) of the gage_df to process.
    :type start_row: int
    :param end_row: The ending row index (exclusive) of the gage_df to process.
    :type end_row: int
    """
    # Function that calculates the closest weather stations to gage and stores in JSON
    # Base u
    for row in range(start_row, end_row):
        gage_info = {}
        gage_info["river_id"] = int(gage_df.iloc[row]['id'])
        gage_lat = gage_df.iloc[row]['latitude']
        gage_long = gage_df.iloc[row]['logitude']
        gage_info["stations"] = []
        total = len(station_df.index)
        for i in range(0, total):
            stat_row = station_df.iloc[i]
            dist = haversine(stat_row["lon"], stat_row["lat"], gage_long, gage_lat)
            st_id = stat_row['stid']
            gage_info["stations"].append({"station_id": st_id, "dist": dist})
        # This bug was actually only later discovered that it puts further away stations first.
        # However subsequent code was then based on it. So we just use negative
        # indices later on (i.e [-20:])
        gage_info["stations"] = sorted(gage_info['stations'], key=lambda i: i["dist"], reverse=True)
        with open(os.path.join(path_dir, str(gage_info["river_id"]) + "stations.json"), 'w') as w:
            count = 0
            json.dump(gage_info, w)
            if count % 100 == 0:
                print("Currently at " + str(count))
            count += 1


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate the great circle distance between two points on the earth (specified in decimal degrees).

    :param lon1: Longitude of the first point.
    :type lon1: float
    :param lat1: Latitude of the first point.
    :type lat1: float
    :param lon2: Longitude of the second point.
    :type lon2: float
    :param lat2: Latitude of the second point.
    :type lat2: float
    :return: The distance between the two points in kilometers.
    :rtype: float
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def get_weather_data(file_path: str, econet_gages: Set[str], base_url: str) -> Dict:
    """Checks the 20 closest stations for a gage to see if weather data is available, either from ASOS (via URL) or ECONet (via local set).

    Updates the gage metadata with the category ('ASOS' or 'ECO').

    :param file_path: Path to the JSON file containing gage and closest station metadata.
    :type file_path: str
    :param econet_gages: A set of known ECONet station IDs.
    :type econet_gages: Set[str]
    :param base_url: The base URL for checking ASOS data availability.
    :type base_url: str
    :return: The updated dictionary containing gage metadata.
    :rtype: Dict
    """
    # Base URL
    # "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={}&data=tmpf&data=p01m&year1=2019&month1=1&day1=1&year2=2019&month2=1&day2=2&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"

    gage_meta_info = {}

    with open(file_path) as f:
        gage_data = json.load(f)
    gage_meta_info["gage_id"] = gage_data["river_id"]
    gage_meta_info["stations"] = []
    # Processes the 20 closest stations (which are the last 20 due to initial sorting bug)
    closest_stations = gage_data["stations"][-20:]
    for station in reversed(closest_stations):
        url = base_url.format(station["station_id"])
        response = requests.get(url)
        if len(response.text) > 100:
            print(response.text)
            gage_meta_info["stations"].append({"station_id": station["station_id"],
                                               "dist": station["dist"], "cat": "ASOS"})
        elif station["station_id"] in econet_gages:
            gage_meta_info["stations"].append({"station_id": station["station_id"],
                                               "dist": station["dist"], "cat": "ECO"})
    return gage_meta_info


def format_dt(date_time_str: str) -> datetime:
    """Converts a date-time string to a datetime object and rounds up to the next hour if minutes are not zero.

    :param date_time_str: The date and time string in "%Y-%m-%d %H:%M" format.
    :type date_time_str: str
    :return: The properly formatted and rounded datetime object.
    :rtype: datetime
    """
    proper_datetime = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")
    if proper_datetime.minute != 0:
        proper_datetime = proper_datetime + timedelta(hours=1)
        proper_datetime = proper_datetime.replace(minute=0)
    return proper_datetime


def convert_temp(temparature: str) -> float:
    """Converts a temperature string to float, using 50 to fill missing values ('M').

    :param temparature: The temperature value as a string (can be a number or 'M').
    :type temparature: str
    :return: The temperature as a float, or 50.0 if missing.
    :rtype: float
    """
    try:
        return float(temparature)
    except BaseException:
        return 50


def process_asos_data(file_path: str, base_url: str) -> Dict:
    """Retrieves ASOS data for stations marked 'ASOS' in the gage metadata, processes it, saves it to a CSV file, and updates the metadata with missing value counts.

    :param file_path: Path to the JSON file containing gage metadata.
    :type file_path: str
    :param base_url: The base URL for retrieving the full ASOS dataset.
    :type base_url: str
    :return: The updated dictionary containing gage metadata.
    :rtype: Dict
    """
    with open(file_path) as f:
        gage_data = json.load(f)
        for station in gage_data["stations"]:
            if station["cat"] == "ASOS":
                response = requests.get(base_url.format(station["station_id"]))
                with open("temp_weather_data.csv", "w+") as f:
                    f.write(response.text)
                df, missing_precip, missing_temp = process_asos_csv("temp_weather_data.csv")
                station["missing_precip"] = missing_precip
                station["missing_temp"] = missing_temp
                df.to_csv(str(gage_data["gage_id"]) + "_" + str(station["station_id"]) + ".csv")
    with open(file_path, "w") as f:
        json.dump(gage_data, f)
    return gage_data


def process_asos_csv(path: str) -> Tuple[pd.DataFrame, int, int]:
    """Processes a raw ASOS CSV file, imputes missing values, and aggregates data hourly.

    Missing precipitation ('M') is filled with 0. Missing temperature ('M') is filled using bi-directional fill and averaging.

    :param path: Path to the temporary ASOS CSV file.
    :type path: str
    :return: A tuple containing the processed DataFrame, the count of missing precipitation values, and the count of missing temperature values.
    :rtype: Tuple[pd.DataFrame, int, int]
    """
    df = pd.read_csv(path)
    missing_precip = df['p01m'][df['p01m'] == 'M'].count()
    missing_temp = df['tmpf'][df['tmpf'] == 'M'].count()
    df['hour_updated'] = df['valid'].map(format_dt)
    df['tmpf'] = pd.to_numeric(df['tmpf'], errors='coerce')

    df['p01m'] = pd.to_numeric(df['p01m'], errors='coerce')
    # Originally the idea for imputation was to
    # replace missing values with an average of the two closest values
    # But since ASOS stations record at different intervals this could
    # actually cause an overestimation of precip. Instead for now we are replacing with 0
    # df['p01m']=(df['p01m'].fillna(method='ffill') + df['p01m'].fillna(method='bfill'))/2
    df['p01m'] = df['p01m'].fillna(0)
    df['tmpf'] = (df['tmpf'].fillna(method='ffill') + df['tmpf'].fillna(method='bfill')) / 2
    df = df.groupby(by=['hour_updated'], as_index=False).agg(
        {'p01m': 'sum', 'valid': 'first', 'tmpf': 'mean'})
    return df, int(missing_precip), int(missing_temp)
