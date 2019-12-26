import unittest
import os
import pandas as pd 
from flood_forecast.preprocessing.process_usgs import process_intermediate_csv
from flood_forecast.preprocessing.interpolate_preprocess import fix_timezones, interpolate_missing_values
class DataQualityTests(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"test_data")

    def test_intermediate_csv(self): 
        df = pd.read_csv(os.path.join(self.test_data_path, "big_black_test_small.csv"))
        result_df, max_flow, min_flow = process_intermediate_csv(df)
        self.assertEqual(result_df.iloc[1]['datetime'].hour, 5)
        self.assertGreater(max_flow, 2640)
        self.assertLess(min_flow,  1600)

    def test_tz_interpolate_fix(self):
        """
        Additional function to test interpolation
        """
        file_path = os.path.join(self.test_data_path, "river_test_sm.csv")
        revised_df = fix_timezones(file_path)
        self.assertEqual(revised_df.iloc[0]['cfs'], 0.0)
        self.assertEqual(revised_df.iloc[1]['tempf'], 19.4)
        revised_df = interpolate_missing_values(revised_df)
        self.assertEqual(0, sum(pd.isnull(revised_df['cfs'])))
        self.assertEqual(0, sum(pd.isnull(revised_df['precip'])))

    def test_chunking(self):
        self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main()

