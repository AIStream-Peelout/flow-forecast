from flood_forecast.preprocessing.pytorch_loaders import CSVSeriesIDLoader
import unittest
import pandas as pd
import os


class TestInterpolationCSVLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = CSVSeriesIDLoader("shit", {}, "")

    def test_seriesid(self):
        print("D")

if __name__ == '__main__':
    unittest.main()
