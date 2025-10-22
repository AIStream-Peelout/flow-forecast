# import json
import pandas as pd


def make_gage_data_csv(file_path: str):
    """Reads a JSON file, converts it into a pandas DataFrame, and sets the index name to 'id'.

    This function is typically used to process metadata about river gages.

    :param file_path: The path to the input JSON file containing gage data.
    :type file_path: Union[str, TextIO]
    :return: A DataFrame where the original JSON keys are the index (named 'id') and the values are columns.
    :rtype: pd.DataFrame
    """
    "returns df"
    with open(file_path) as f:
        df = pd.read_json(f)
        df = df.T
        df.index.name = "id"
        return df

# todo define this function properly (what is econet?)
# def make_station_meta(file_path_eco: str, file_path_assos: str):
#     core_columns = econet[['Station', 'Name', 'Latitude', 'Longitude',
#                            'Elevation', 'First Ob', 'Supported By', 'Time Interval(s)', 'Precip']]

# todo define this function properly (haversine is not defined)
# def get_closest_gage_list(station_df: pd.DataFrame, gage_df: pd.DataFrame):
#     for row in gage_df.iterrows():
#         gage_info = {}
#         gage_info["river_id"] = row[1]['id']
#         gage_lat = row[1]['latitude']
#         gage_long = row[1]['logitude']
#         gage_info["stations"] = []
#         for stat_row in station_df.iterrows():
#             dist = haversine(stat_row[1]["lon"], stat_row[1]["lat"], gage_long, gage_lat)
#             st_id = stat_row[1]['stid']
#             gage_info["stations"].append({"station_id": st_id, "dist": dist})
#         gage_info["stations"] = sorted(gage_info['stations'], key=lambda i: i["dist"], reverse=True)
#         print(gage_info)
#         with open(str(gage_info["river_id"]) + "stations.json", 'w') as w:
#             json.dump(gage_info, w)
