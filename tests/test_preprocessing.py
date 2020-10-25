from flood_forecast.preprocessing.interpolate_preprocess import back_forward_generic
import unittest
import pandas as pd
import os


class TestInterpolationCode(unittest.TestCase):
    def setUp(self):
        file_path = os.path.join(os.path.dirname(__file__), "test_data", "farm_ex.csv")
        self.df = pd.read_csv(file_path)

    def test_back_forward(self):
        df = back_forward_generic(self.df, ["NumberOfAnimals"])
        self.assertEqual(df.iloc[3]["NumberOfAnimals"], 165)

    def test_make_function(self):
        pass

if __name__ == '__main__':
    unittest.main()
