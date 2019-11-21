from math import radians, cos, sin, asin, sqrt
import pandas as pd
import os
import json
from typing import Set, Tuple, Dict
import requests
from datetime import datetime, timedelta

def get_closest_gage(gage_df:pd.DataFrame, station_df:pd.DataFrame, path_dir:str, start_row:int, end_row:int):
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
      gage_info["stations"].append({"station_id":st_id, "dist":dist})
    gage_info["stations"] = sorted(gage_info['stations'], key = lambda i: i["dist"], reverse=True) 
    with open(os.path.join(path_dir, str(gage_info["river_id"]) + "stations.json"), 'w') as w:
      json.dump(gage_info, w)
      if count%100 == 0:
        print("Currently at " + str(count))
      count +=1 
      
def get_weather_data(file_path:str, econet_gages:Set, base_url:str):
  """
  Function that retrieves if station has weather 
  data for a specific gage either from ASOS or ECONet 
  """
  # Base URL "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={}&data=tmpf&data=p01m&year1=2019&month1=1&day1=1&year2=2019&month2=1&day2=2&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"

  gage_meta_info = {}
  
  with open(file_path) as f:
    gage_data = json.load(f)
  gage_meta_info["gage_id"] = gage_data["river_id"]
  gage_meta_info["stations"] = []
  closest_stations = gage_data["stations"][-20:]
  for station in reversed(closest_stations):
    url = base_url.format(station["station_id"])
    response = requests.get(url)
    if len(response.text)>100:
      print(response.text)
      gage_meta_info["stations"].append({"station_id":station["station_id"], 
                                         "dist":station["dist"], "cat":"ASOS"})
    elif station["station_id"] in econet_gages:
      gage_meta_info["stations"].append({"station_id":station["station_id"], 
                                         "dist":station["dist"], "cat":"ECO"})
  return gage_meta_info

def format_dt(date_time_str:str) -> datetime:
  proper_datetime = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")
  if proper_datetime.minute != 0:
    proper_datetime = proper_datetime + timedelta(hours=1)
    proper_datetime = proper_datetime.replace(minute=0)
  return proper_datetime
  
def convert_temp(temparature:str) -> float:
  """
  Note here temp could be a number or 'M'
  which stands for missing. We use 50 at the moment 
  to fill missing values. 
  """
  try: 
    return float(temparature)
  except:
    return 50

def handle_missing_precip(precip:float, median:float) -> float:
  if precip=='M':
    return median
  return precip

def process_asos_data(file_path:str, base_url:str):
  """
  Function that saves the ASOS data to CSV 
  uses output of get weather data.
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
      df.to_csv(str(gage_data["gage_id"]) + "_" + str(station["station_id"])+".csv")

def process_asos_csv(path:str):
    df = pd.read_csv(path)
    missing_precip = df['p01m'][df['p01m']=='M'].count()
    missing_temp = df['tmpf'][df['tmpf']=='M'].count()
    df['hour_updated'] = df['valid'].map(format_dt)
    df['tmpf'] = df['tmpf'].map(convert_temp)
    median = df['p01m'][df['p01m']!='M'].median()
    # TODO use average of preceeding and subsequent non-missing value
    df['p01m'] = df['p01m'].map(lambda x: handle_missing_precip(x, median))
    df = df.groupby(by=['hour_updated'], as_index=False).agg({'p01m': 'sum', 'valid': 'first', 'tmpf': 'mean'})
    return df, missing_precip, missing_temp

