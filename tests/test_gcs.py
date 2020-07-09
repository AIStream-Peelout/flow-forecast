import unittest
from flood_forecast.gcp_integration import download_file
import pandas as pd
import os
import json
from datetime import datetime
import pytz


class JoinTestGCS(unittest.TestCase):
    def setUp(self):
        self.client = storage.Client()

    def test_download_file(self):
        download_file("predict_cfs", "experiments/", self.client, "example_config.json")

if __name__ == '__main__':
    unittest.main()
