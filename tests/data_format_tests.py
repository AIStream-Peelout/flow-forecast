from flood_forecast.preprocessing.closest_station import get_weather_data, get_closest_gage, format_dt, convert_temp, process_asos_csv
from datetime import datetime
import unittest

class DataQualityTests(unittest.TestCase):
    def test_format_dt(self):
        self.assertEqual(format_dt("2017-04-07 08:55"), datetime(year=2017, month=4, day=7, hour=9))
        self.assertEqual(format_dt("2018-04-08 23:55"), datetime(year=2018, month=4, day=9, hour=0)) 

    def test_process_asos_csv(self):
        df = process_asos_csv("small_test.csv")
        self.assertEqual(df.illoc[0]['p01m'], 92)
        self.assertEqual(df.illoc[0]['tmpf'], 52.5)
        self.assertEqual(df.illoc[0]['hour_updated'].hour, 1)
    