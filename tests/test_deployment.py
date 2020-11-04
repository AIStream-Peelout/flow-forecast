import os
import json
from flood_forecast.deployment.inference import load_model, InferenceMode
import unittest
from datetime import datetime


class InferenceTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")) as y:
            self.config_test = json.load(y)
        self.new_csv_path = "gs://task_ts_data/Massachusetts_Middlesex_County.csv"
        self.weight_path = "gs://coronaviruspublicdata/experiments/01_July_202009_44PM_model.pth"
        self.infer_class = InferenceMode(20, 30, self.config_test, self.new_csv_path, self.weight_path)
        os.environ["infer_bucket"] = "task_ts_data"

    def test_load_model(self):
        load_model(self.config_test, self.new_csv_path, self.weight_path)

    def test_infer_mode(self):
        self.infer_class.infer_now(datetime(2020, 5, 5), self.new_csv_path, "temp.csv", "inference_files/new_infer.csv")

    def test_plot_model(self):
        pass

if __name__ == "__main__":
    unittest.main()
