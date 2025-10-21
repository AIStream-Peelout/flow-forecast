from flood_forecast.preprocessing.interpolate_preprocess import back_forward_generic
from flood_forecast.preprocessing.temporal_feats import feature_fix
import unittest
import pandas as pd
import os


class TestInterpolationCode(unittest.TestCase):
    def setUp(self) -> None:
        """
        Sets up test data by loading CSV files into pandas DataFrames.
        
        :return: None
        :rtype: None
        """

        file_path = os.path.join(os.path.dirname(__file__), "test_data", "farm_ex.csv")
        file_path_2 = os.path.join(os.path.dirname(__file__), "test_data", "fake_test_small.csv")
        self.df = pd.read_csv(file_path)
        self.df_2 = pd.read_csv(file_path_2, delimiter="\t")
        self.df_2["datetime"] = pd.to_datetime(self.df_2["datetime"])

    def test_back_forward(self) -> None:
        """
        Tests the `back_forward_generic` interpolation function on sample data.
        Verifies the correctness of interpolated values.
        
        :return: None
        :rtype: None
        """

        """Test the generation of forward and backward data interp."""
        df = back_forward_generic(self.df, ["NumberOfAnimals"])
        self.assertEqual(df.iloc[3]["NumberOfAnimals"], 165)

    def test_make_temp_feats(self) -> None:
        """
        Tests `feature_fix` function with cyclical 'hour' datetime parameter.
        Checks if expected sine and cosine features, as well as normalization, are present.
        
        :return: None
        :rtype: None
        """

        feats = feature_fix({"datetime_params": {"hour": "cyclical"}}, "datetime", self.df_2)
        self.assertIn("sin_hour", feats[0].columns)
        self.assertIn("cos_hour", feats[0].columns)
        self.assertIn("norm", feats[0].columns)

    def test_make_temp_feats2(self) -> None:
        """
        Tests `feature_fix` function with numerical 'year' and cyclical 'day' datetime parameters.
        Verifies presence of year, sine and cosine day features, and normalization column.
        
        :return: None
        :rtype: None
        """

        feats = feature_fix({"datetime_params": {"year": "numerical", "day": "cyclical"}}, "datetime", self.df_2)
        self.assertIn("year", feats[0].columns)
        self.assertIn("sin_day", feats[0].columns)
        self.assertIn("cos_day", feats[0].columns)
        self.assertIn("norm", feats[0].columns)

if __name__ == '__main__':
    unittest.main()
