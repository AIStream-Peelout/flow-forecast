from math import radians, cos, sin, asin, sqrt
import pandas as pd
import os
import json

def get_closest_gage(gage_df:pd.DataFrame, station_df:pd.DataFrame, path_dir:str):
  for row in gage_df.iterrows():
      gage_info = {}
      gage_info["river_id"] = row[1]['id']
      gage_lat = row[1]['latitude']
      gage_long = row[1]['logitude']
      gage_info["stations"] = []
      min_distance = 4000000

      for stat_row in station_df.iterrows():
            dist = haversine(stat_row[1]["lon"], stat_row[1]["lat"], gage_long, gage_lat)
            st_id = stat_row[1]['stid']
            gage_info["stations"].append({"station_id":st_id, "dist":dist})
      gage_info["stations"] = sorted(gage_info['stations'], key = lambda i: i["dist"], reverse=True) 
      with open(os.path.join(path_dir, str(gage_info["river_id"]) + "stations.json"), 'w') as w:
        json.dump(gage_info, w)
