import sys
sys.path.append("..")
from flood_forecast.preprocessing.preprocess_da_rnn import TrainData, format_data, make_data 
import unittest
import pandas as pd
import os 
class TestPreprocessingDA(unittest.TestCase):
    def test_format_data(self):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_data", "test_format_data.csv"))
        self.assertEqual(type(format_data(df, ["height"])), TrainData)
        self.assertEqual(len(format_data(df, ["height"]).feats[0]), 2)

    def test_make_function(self):
        result = make_data(os.path.join(os.path.dirname(__file__), "test_data", "test_format_data.csv"), "height", 3)
        self.assertEqual(len(result.feats), 1)
        self.assertEqual(len(result.targs), 1)

if __name__ == '__main__':
    unittest.main()