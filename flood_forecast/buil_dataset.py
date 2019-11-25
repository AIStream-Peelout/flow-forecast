import os
from flood_forecast.preprocessing.closest_station import get_weather_data
from flood_forecast.eco_gage_set import eco_gage_set
from typing import Set

def build_dataset(json_full_path:str, asos_base_url:str, econet_data:Set, start:int, end_index:int):
  directory = os.fsencode(json_full_path)
  directory_list = sorted(os.listdir(directory))
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    get_weather_data(os.path.join(json_full_path, filename), econet_data, asos_base_url)

def get_eco_netset(directory_path):
    """
    Econet data was supplied to us by the NC State climate office. They gave
    us a directory of CSV files in following format `LastName_First_station_id_Hourly.txt`
    This code simply constructs a set of stations based on what is in the folder.
    """
    pass 
