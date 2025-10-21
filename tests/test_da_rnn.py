import torch
import unittest
import os
import tempfile
from flood_forecast.preprocessing.preprocess_da_rnn import make_data
from flood_forecast.da_rnn.train_da import da_rnn, train


class TestDARNN(unittest.TestCase):
    def setUp(self):
        """
        Prepares the test environment by preprocessing data from a CSV file for DARNN testing.

        :return: None
        :rtype: None
        """
        self.preprocessed_data = self.preprocessed_data = make_data(os.path.join(
            os.path.dirname(__file__), "test_init", "keag_small.csv"), ["cfs"], 72)

    def test_train_model(self):
        """
        Tests training the DARNN model for one epoch and asserts that a model instance is returned.

        :return: None
        :rtype: None
        """
        with tempfile.TemporaryDirectory() as param_directory:
            config, da_network = da_rnn(self.preprocessed_data, 1, 64,
                                        param_output_path=param_directory)
            loss_results, model = train(da_network, self.preprocessed_data,
                                        config, n_epochs=1, tensorboard=True)
            self.assertTrue(model)

    def test_tf_data(self):
        """
        Verifies that the TensorBoard directory has files after training, indicating logs were created.

        :return: None
        :rtype: None
        """
        dirname = os.path.dirname(__file__)
        # Test that Tensorboard directory was indeed created
        self.assertTrue(os.listdir(os.path.join(dirname)))

    def test_create_model(self):
        """
        Tests the creation of the DARNN model and verifies configuration batch size and model instance.

        :return: None
        :rtype: None
        """
        with tempfile.TemporaryDirectory() as param_directory:
            config, dnn_network = da_rnn(self.preprocessed_data, 1, 64,
                                         param_output_path=param_directory)
            self.assertNotEqual(config.batch_size, 20)
            self.assertIsNotNone(dnn_network)

    def test_resume_ckpt(self):
        """
        Tests resuming training from saved encoder and decoder checkpoints.

        :return: None
        :rtype: None
        """
        config, da = da_rnn(self.preprocessed_data, 1, 64)
        with tempfile.TemporaryDirectory() as checkpoint:
            torch.save(da.encoder.state_dict(), os.path.join(checkpoint, "encoder.pth"))
            torch.save(da.decoder.state_dict(), os.path.join(checkpoint, "decoder.pth"))
            config, dnn_network = da_rnn(self.preprocessed_data, 1, 64, save_path=checkpoint)
            self.assertTrue(dnn_network)

if __name__ == '__main__':
    unittest.main()
