import torch
import unittest
import os
import tempfile
from flood_forecast.preprocessing.preprocess_da_rnn import make_data
from flood_forecast.da_rnn.train_da import da_rnn, train


class TestDARNN(unittest.TestCase):
    def setUp(self):
        self.preprocessed_data = self.preprocessed_data = make_data(os.path.join(
            os.path.dirname(__file__), "test_init", "keag_small.csv"), ["cfs"], 72)

    def test_train_model(self):
        with tempfile.TemporaryDirectory() as param_directory:
            config, da_network = da_rnn(self.preprocessed_data, 1, 64,
                                        param_output_path=param_directory)
            loss_results, model = train(da_network, self.preprocessed_data,
                                        config, n_epochs=1, tensorboard=True)
            self.assertTrue(model)

    def test_tf_data(self):
        dirname = os.path.dirname(__file__)
        # Test that Tensorboard directory was indeed created
        self.assertTrue(os.listdir(os.path.join(dirname)))

    def test_create_model(self):
        with tempfile.TemporaryDirectory() as param_directory:
            config, dnn_network = da_rnn(self.preprocessed_data, 1, 64,
                                         param_output_path=param_directory)
            self.assertNotEqual(config.batch_size, 20)
            self.assertIsNotNone(dnn_network)

    def test_resume_ckpt(self):
        """ This test is dependent on test_train_model succeding"""
        config, da = da_rnn(self.preprocessed_data, 1, 64)
        with tempfile.TemporaryDirectory() as checkpoint:
            torch.save(da.encoder.state_dict(), os.path.join(checkpoint, "encoder.pth"))
            torch.save(da.decoder.state_dict(), os.path.join(checkpoint, "decoder.pth"))
            config, dnn_network = da_rnn(self.preprocessed_data, 1, 64, save_path=checkpoint)
            self.assertTrue(dnn_network)

if __name__ == '__main__':
    unittest.main()
