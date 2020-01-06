import unittest
import os 
from datetime import datetime
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader

class DataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"test_data")
        data_base_params = {"forecast_history": 20, "class":"default", "forecast_length":20, "relevant_cols":["cfs", "temp", "precip"], "target_col":["cfs"], "interpolate": False}
        self.test_loader = CSVTestLoader(os.path.join(self.test_data_path, "keag_small.csv"), 336, **data_base_params)

    def test_loader2_get_item(self):
        self.test_loader[0] 
    
if __name__ == '__main__':
    unittest.main()
    


   
    