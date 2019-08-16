import sys
sys.path.append("..")
from flood_forecast.preprocessing.preprocess_da_rnn import TrainData, format_data, make_data 
import unittest
import pandas as pd
class TestPreprocessingDA(unittest.TestCase):
    def test_format_data(self):
        df = pd.read_csv("test_format_data.csv")
        self.assertEqual(type(format_data(df, ["height"])), TrainData)
        self.assertEqual(len(format_data(df, ["height"]).feats[0]), 2)
    def test_make_data(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()