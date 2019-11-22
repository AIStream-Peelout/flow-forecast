from flood_forecast.preprocessing.closest_station import get_weather_data, get_closest_gage, format_dt, convert_temp, process_asos_csv
from datetime import datetime
import unittest
import os 

class DataQualityTests(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"test_data")

    def test_format_dt(self):
        self.assertEqual(format_dt("2017-04-07 08:55"), datetime(year=2017, month=4, day=7, hour=9))
        self.assertEqual(format_dt("2018-04-08 23:55"), datetime(year=2018, month=4, day=9, hour=0)) 

    def test_convert_temp(self):
        self.assertEqual(convert_temp("50.3"), 50.3)
        self.assertEqual(convert_temp("-12.0"), -12.0)
        self.assertEqual(convert_temp("M"), 50)

    def test_process_asos_csv(self):
        df, precip_missing, temp_missing = process_asos_csv(os.path.join( self.test_data_path,"small_test.csv"))
        self.assertEqual(df.iloc[1]['p01m'], 47)
        self.assertEqual(df.iloc[0]['tmpf'], 50)
        self.assertEqual(df.iloc[1]['hour_updated'].hour, 1)
        self.assertEqual(df.iloc[1]['tmpf'], 53)
        self.assertEqual(precip_missing, 0)
        self.assertEqual(temp_missing, 0)
    
    def test_process_asos_full(self):
        df, precip_missing, temp_missing = process_asos_csv(os.path.join( self.test_data_path, "asos_raw.csv"))
        self.assertGreater(temp_missing,10)
        self.assertGreater(precip_missing, 2)

    def test_value_imputation(self):
        df, precip_missing, temp_missing = process_asos_csv(os.path.join(self.test_data_path, "imputation_test.csv"))
        self.assertEqual(df.iloc[0]['p01m'], 0)
        #self.assertEqual(df.iloc[2]['p01m'], 1)

if __name__ == '__main__':
    unittest.main()
    


   
    