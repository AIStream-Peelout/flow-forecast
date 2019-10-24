import json
import pandas as pd 

def make_gage_data_csv(file_path:str):
    with open(file_path) as f: 
        df = pd.read_json(f)
        df = df.T
        df.index.name = "id"
        return df