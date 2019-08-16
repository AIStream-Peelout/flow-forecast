import sys
sys.path.append("..")
from flood_forecast.preprocessing.preprocess_da_rnn import TrainData, format_data, make_data 
from flood_forecast.da_rnn.train_da import da_rnn, train
import unittest
import pandas as pd
class TestPreprocessingDA(unittest.TestCase):
    def setUp(self):
        self.preprocessed_data = make_data("test_data/kenduskeag_small.csv", "cfs")

    def test_train_model(self):
        da = da_rnn(self.preprocessed_data, 1, 64)
        train(da, self.preprocessed_data, )
        self.assertEqual(1,1)
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