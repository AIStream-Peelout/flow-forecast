from flood_forecast.preprocessing.closest_station import get_weather_data, get_closest_gage, format_dt
from datetime import datetime
import unittest

class DataQualityTests(unittest.TestCase):
    def test_format_dt(self):
        self.assertEqual(format_dt("2017-04-07 08:55"), datetime(year=2017, month=4, day=7, hour=9))
        self.assertEqual(format_dt("2018-04-08 23:55"), datetime(year=2018, month=4, day=9, hour=0))
        