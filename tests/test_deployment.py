import os
import json
from flood_forecast.deployment.inference import load_model
import unittest


class InferenceTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")) as y:
            self.config_test = json.load(y)
        self.new_csv_path = "gs://task_ts_data/Massachusetts_Middlesex_County.csv"
        self.weight_path = "gs://coronaviruspublicdata/experiments/01_July_202009_44PM_model.pth"

    def test_load_model(self):
        load_model(self.config_test, self.new_csv_path, self.weight_path)

if __name__ == "__main__":
    unittest.main()
