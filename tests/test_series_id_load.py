import unittest
from flood_forecast.preprocessing.pytorch_loaders import CSVSeriesIDLoader


class SeriesIDLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = CSVSeriesIDLoader()
