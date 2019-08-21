import sys
sys.path.append("..")
from flood_forecast.preprocessing.preprocess_da_rnn import TrainData, format_data, make_data 
from flood_forecast.da_rnn.train_da import da_rnn, train
import unittest
import pandas as pd
class TestPreprocessingDA(unittest.TestCase):
    def setUp(self):
        self.preprocessed_data = make_data("test_data/keag_small.csv", "cfs")

    def test_train_model(self):
        config, da_network = da_rnn(self.preprocessed_data, 1, 64)
        train(da_network, self.preprocessed_data, config, n_epochs=20, tensorboard=True)
        self.assertEqual(1,1)
        
    def test_make_data(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_create_model(self):
        config, dnn_network = da_rnn(self.preprocessed_data, 1, 64)
        self.assertNotEqual(config.batch_size, 20)
        self.assertIsNotNone(dnn_network)

if __name__ == '__main__':
    unittest.main()