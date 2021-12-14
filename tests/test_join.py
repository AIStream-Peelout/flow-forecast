import unittest
from flood_forecast.preprocessing.buil_dataset import combine_data
import pandas as pd
import os
import json
from datetime import datetime
import pytz


class JoinTest(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        # 1

    def test_join_function(self):
        df = pd.read_csv(os.path.join(self.test_data_path, "fake_test_small.csv"), sep="\t")
        asos_df = pd.read_csv(os.path.join(self.test_data_path, "asos-12N_small.csv"))
        old_timezone = pytz.timezone("America/New_York")
        new_timezone = pytz.timezone("UTC")
        # This assumes timezones are consistent throughout the USGS stream (this should be true for all)
        df["datetime"] = df["datetime"].map(lambda x: old_timezone.localize(
            datetime.strptime(x, "%Y-%m-%d %H:%M")).astimezone(new_timezone))
        with open(os.path.join(self.test_data_path, "big_black_md.json")) as a:
            meta_data = json.load(a)
            self.assertEqual(meta_data['gage_id'], 1010070)
        result_df, nan_f, nan_p = combine_data(df, asos_df)
        self.assertEqual(result_df.iloc[0]['p01m'], 0)
        self.assertEqual(result_df.iloc[0]['cfs'], 2210)
        self.assertEqual(result_df.iloc[0]['tmpf'], 19.94)

if __name__ == '__main__':
    unittest.main()
