import pandas as pd 
import requests 
from datetime import datetime
from typing import Tuple, Dict
# url format 
def make_usgs_data(start_date:datetime, end_date:datetime, site_number:str):
    base_url = "https://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&cb_00065&format=rdb&"
    full_url = base_url + "site_no=" + site_number + "&period=&begin_date="+start_date.strftime("%Y-%m-%d") + "&end_date="+end_date.strftime("%Y-%m-%d")
    print("Getting request from USGS")
    print(full_url)
    r = requests.get(full_url)
    with open(site_number + ".txt", "w") as f:
        f.write(r.text)
    print("Request finished")
    response_data = process_response_text(site_number + ".txt")
    create_csv(response_data[0], response_data[1], site_number)
    return pd.read_csv(site_number + "_flow_data.csv")

def process_response_text(file_name:str)->Tuple[str, Dict]:
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
                if len(the_split_line)<2:
                    params = False 
                else:
                    extractive_params[the_split_line[0]+"_"+the_split_line[1]] = df_label(the_split_line[2])
            if len(the_split_line)>2:
                if the_split_line[0]== "TS":
                    params = True
            i+=1
        with open(file_name.split(".")[0] + "data.tsv", "w") as t:
            t.write("".join(lines[i:]))
        return file_name.split(".")[0] + "data.tsv", extractive_params

def df_label(usgs_text:str)->str:
    usgs_text = usgs_text.replace(",","")
    if usgs_text == "Discharge":
        return "cfs"
    elif usgs_text=="Gage":
        return "height"
    else:
        return usgs_text

def create_csv(file_path:str, params_names:dict, site_number:str):
    """
    Function that creates the final version of the CSV file
    """
    df = pd.read_csv(file_path, sep="\t")
    for key, value in params_names.items():
        df[value] = df[key]
    df.to_csv(site_number + "_flow_data.csv")
