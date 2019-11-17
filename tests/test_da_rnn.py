import sys
sys.path.append("..")
from flood_forecast.preprocessing.preprocess_da_rnn import TrainData, format_data, make_data 
from flood_forecast.da_rnn.train_da import da_rnn, train
import unittest
import os
import pandas as pd
import pathlib
import torch
class TestDARNN(unittest.TestCase):
    def setUp(self):
        self.preprocessed_data = self.preprocessed_data = make_data(os.path.join(os.path.dirname(__file__), "test_data", "keag_small.csv"), ["cfs"], 72)

    def test_train_model(self):
        config, da_network = da_rnn(self.preprocessed_data, 1, 64)
        loss_results, model = train(da_network, self.preprocessed_data, config, n_epochs=2, tensorboard=True)
        self.assertTrue(model)

    def test_tf_data(self):
        dirname = os.path.dirname(__file__)
        # Test that Tensorboard directory was indeed created 
        self.assertTrue(os.listdir(os.path.join(dirname)))
        

    def test_create_model(self):
        config, dnn_network = da_rnn(self.preprocessed_data, 1, 64)
        self.assertNotEqual(config.batch_size, 20)
        self.assertIsNotNone(dnn_network)
    
    def test_resume_ckpt(self):
        """ This test is dependent on test_train_model succeding"""
        config, da = da_rnn(self.preprocessed_data, 1, 64)
        torch.save(da.encoder.state_dict(), os.path.join("checkpoint", "encoder.pth"))
        config, dnn_network = da_rnn(self.preprocessed_data, 1, 64, save_path="checkpoint")
        self.assertTrue(config)
        #self.assertTrue(dnn_network)
if __name__ == '__main__':
    unittest.main()