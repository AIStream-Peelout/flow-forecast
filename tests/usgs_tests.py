import unittest
import os
import pandas as pd 
from flood_forecast.preprocessing.process_usgs import process_intermediate_csv
class DataQualityTests(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"test_data")

    def test_intermediate_csv(self): 
        df = pd.read_csv(os.path.join(self.test_data_path, "big_black_test_small.csv"))
        result_df, max_flow, min_flow = process_intermediate_csv(df)
        self.assertEqual(result_df.iloc[1]['datetime'].hour, 5)
        self.assertGreater(max_flow, 2640)
        self.assertLess(min_flow,  1600)

