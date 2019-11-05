from math import radians, cos, sin, asin, sqrt
import pandas as pd
import os
import json

def get_closest_gage(gage_df:pd.DataFrame, station_df:pd.DataFrame, path_dir:str, start_row:int, end_row:int):
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
      gage_info["stations"].append({"station_id":st_id, "dist":dist})
    gage_info["stations"] = sorted(gage_info['stations'], key = lambda i: i["dist"], reverse=True) 
    with open(os.path.join(path_dir, str(gage_info["river_id"]) + "stations.json"), 'w') as w:
      json.dump(gage_info, w)
      if count%100 == 0:
        print("Currently at " + str(count))
      count +=1 
