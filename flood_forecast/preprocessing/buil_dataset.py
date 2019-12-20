import os
from flood_forecast.preprocessing.closest_station import get_weather_data, process_asos_csv
from flood_forecast.eco_gage_set import eco_gage_set
from flood_forecast.make_usgs import make_usgs_data
from typing import Set
import json
from datetime import datetime


def build_weather_csv(json_full_path, asos_base_url, base_url_2, econet_data, visited_gages_path, start=0, end_index=100):
  directory = os.fsencode(json_full_path)
  sorted_list = sorted(os.listdir(directory))
  for i in range(start, end_index):
    file = sorted_list[i]
    filename = os.fsdecode(file)
    get_weather_data(os.path.join(json_full_path, filename), econet_data, asos_base_url, visited_gages_path)
    process_asos_data(os.path.join(json_full_path, filename), base_url_2, visited_gages_path)

def make_usgs(meta_data_path:str, start, end_index):
  meta_directory = os.fsencode(meta_data_path)
  sorted_list = sorted(os.listdir(directory))
  for i in range(start, end_index):
    with open(i) as d:
      data = json.loads(d)
    #make_usgs_data(datetime(2014, 1, 1), datetime(2019,1,1), data["gage_id"])

def join_data(weather_csv, meta_json_file, flow_csv):
  pass

def create_visited():
  visited_gages = {"stations_visited":{}, "saved_complete":{}}
  with open("visited_gages.json", "w+") as f:
    json.dump(visited_gages,f)

def get_eco_netset(directory_path):
  """
  Econet data was supplied to us by the NC State climate office. They gave
  us a directory of CSV files in following format `LastName_First_station_id_Hourly.txt`
  This code simply constructs a set of stations based on what is in the folder.
  """
  eco_gage_set = {"A"}
  directory = os.fsencode(directory_path)
  print(sorted(os.listdir(directory)))
  for file in sorted(os.listdir(directory)):
    filename = os.fsdecode(file)
    try: 
        #print(filename.split("c_")[1].split("_H")[0])
        eco_gage_set.add(filename.split("c_")[1].split("_H")[0])
    except:
        print(filename)
  return eco_gage_set

def create_usgs(meta_data_dir):
  for file_name in os.listdir(meta_data_dir):
    gage_id = file_name.split("stations")[0]
    with open(os.path.join(meta_data_dir , file_name)) as f:
      #data = json.load(f)
      pass 
    make_usgs_data(datetime(2014, 1, 1), datetime(2019,1,1), "0"+gage_id)