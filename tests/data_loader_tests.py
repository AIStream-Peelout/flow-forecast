import unittest
import os 
import torch
import pandas as pd
from datetime import datetime
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader

class DataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"test_data")
        data_base_params = {"history_length": 20, "class":"default", "forecast_length":20, "relevant_cols":["cfs", "temp", "precip"], "target_col":["cfs"], "interpolate_param": False}
        self.test_loader = CSVTestLoader(os.path.join(self.test_data_path, "keag_small.csv"), 336, **data_base_params)

    def test_loader2_get_item(self):
        src, forecast_start_index, df = self.test_loader[0]  
        self.assertEqual(type(src), torch.Tensor)
        self.assertEqual(forecast_start_index, 20)
        self.assertEqual(type(df), pd.Dataframe)
    
if __name__ == '__main__':
    unittest.main()
    


   
    