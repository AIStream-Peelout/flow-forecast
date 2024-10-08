import unittest
import os
import pandas as pd
from flood_forecast.preprocessing.process_usgs import process_intermediate_csv
from flood_forecast.preprocessing.interpolate_preprocess import fix_timezones
from flood_forecast.preprocessing.interpolate_preprocess import interpolate_missing_values


class DataQualityTests(unittest.TestCase):
    def setUp(self):
        # These are historical tests.
        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

    def test_intermediate_csv(self):
        df = pd.read_csv(os.path.join(self.test_data_path, "big_black_test_small.csv"), sep="\t")
        result_df, max_flow, min_flow = process_intermediate_csv(df)
        self.assertEqual(result_df.iloc[1]['datetime'].hour, 6)
        self.assertGreater(max_flow, 2640)
        self.assertLess(min_flow, 1600)

    def test_tz_interpolate_fix(self):
        """Additional function to test data interpolation."""
        file_path = os.path.join(self.test_data_path, "river_test_sm.csv")
        test_d = pd.read_csv(file_path)
        revised_df = fix_timezones(test_d)
        self.assertEqual(revised_df.iloc[0]['cfs'], 0.0)
        self.assertEqual(revised_df.iloc[1]['tmpf'], 19.94)
        revised_df = interpolate_missing_values(revised_df)
        self.assertEqual(0, sum(pd.isnull(revised_df['cfs'])))
        self.assertEqual(0, sum(pd.isnull(revised_df['precip'])))

if __name__ == '__main__':
    unittest.main()
