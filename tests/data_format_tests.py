from flood_forecast.preprocessing.closest_station import get_weather_data, format_dt, convert_temp, \
    process_asos_csv, process_asos_data
from datetime import datetime
import unittest
import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


class DataQualityTests(unittest.TestCase):
    """
    Unit tests for functions involved in preprocessing ASOS weather station data.
    """
    def setUp(self):
        """
        Set up the test environment by defining the test data path.

        :return: None
        :rtype: None
        """
        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

    def test_format_dt(self):
        """
        Test the `format_dt` function for correct datetime conversion and rounding.

        :return: None
        :rtype: None
        """
        self.assertEqual(format_dt("2017-04-07 08:55"), datetime(year=2017, month=4, day=7, hour=9))
        self.assertEqual(format_dt("2018-04-08 23:55"), datetime(year=2018, month=4, day=9, hour=0))

    def test_convert_temp(self):
        """
        Test the `convert_temp` function to handle valid temps and missing values.

        :return: None
        :rtype: None
        """
        self.assertEqual(convert_temp("50.3"), 50.3)
        self.assertEqual(convert_temp("-12.0"), -12.0)
        self.assertEqual(convert_temp("M"), 50)

    def test_process_asos_csv(self):
        """
        Test parsing and processing of a small ASOS CSV file.

        :return: None
        :rtype: None
        """
        df, precip_missing, temp_missing = process_asos_csv(
            os.path.join(self.test_data_path, "small_test.csv"))
        self.assertEqual(df.iloc[1]['p01m'], 47)
        self.assertEqual(df.iloc[0]['tmpf'], 50)
        self.assertEqual(df.iloc[1]['hour_updated'].hour, 1)
        self.assertEqual(df.iloc[1]['tmpf'], 53)
        self.assertEqual(precip_missing, 0)
        self.assertEqual(temp_missing, 0)

    def test_process_asos_full(self):
        """
        Test a larger CSV for correctly identifying missing precipitation and temperature values.

        :return: None
        :rtype: None
        """
        df, precip_missing, temp_missing = process_asos_csv(
            os.path.join(self.test_data_path, "asos_raw.csv"))
        self.assertGreater(temp_missing, 10)
        self.assertGreater(precip_missing, 2)

    def test_value_imputation(self):
        """
        Test handling of missing values and their imputation logic.

        :return: None
        :rtype: None
        """
        df, precip_missing, temp_missing = process_asos_csv(
            os.path.join(self.test_data_path, "imputation_test.csv"))
        self.assertEqual(df.iloc[0]['p01m'], 0)
        self.assertEqual(df.iloc[2]['p01m'], 23)

    @patch("flood_forecast.preprocessing.closest_station.requests.get")
    def test_get_weather_data(self, mock_get):
        """
        Test weather-station classification with deterministic HTTP responses.

        :return: None
        :rtype: None
        """
        url = (
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
            "station={}&data=tmpf&data=p01m&year1=2019&month1=1&day1=1&year2=2019&month2=1&"
            "day2=2&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"
        )
        mock_get.side_effect = lambda requested_url: SimpleNamespace(
            text=("station,valid,tmpf,p01m\n" +
                  "CYCX,2019-01-01 00:00,17.60,0.00\n" * 4
                  if "station=CYCX" in requested_url else "no data"))
        result = get_weather_data(os.path.join(self.test_data_path, "full_out.json"), {}, url)
        self.assertEqual(result["gage_id"], 1021200)
        self.assertEqual([station["station_id"] for station in result["stations"]], ["CYCX"])
        self.assertEqual(mock_get.call_count, 20)

    @patch("flood_forecast.preprocessing.closest_station.pd.DataFrame.to_csv")
    @patch("flood_forecast.preprocessing.closest_station.process_asos_csv")
    @patch("flood_forecast.preprocessing.closest_station.download_asos_csv")
    def test_process_asos_data(self, mock_download, mock_process, _mock_to_csv):
        """
        Full processing test for ASOS weather data using local JSON input and validating output structure.

        :return: None
        :rtype: None
        """
        full_data_url = (
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
            "station={}&data=tmpf&data=p01m&year1=2014&month1=1&day1=1&year2=2019&month2=1&day2=2"
            "&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"
        )
        mock_process.return_value = (pd.DataFrame({"p01m": [0.0], "tmpf": [50.0]}), 3, 4)
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_path = os.path.join(temporary_directory, "asos_process.json")
            shutil.copyfile(os.path.join(self.test_data_path, "asos_process.json"), metadata_path)
            river_result = process_asos_data(metadata_path, full_data_url)
        self.assertGreater(river_result["stations"][1]["missing_temp"], -1)
        self.assertGreater(river_result["stations"][2]["missing_precip"], -1)
        self.assertEqual(mock_download.call_count, len(river_result["stations"]))


if __name__ == '__main__':
    unittest.main()
