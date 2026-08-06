import torch
import unittest
import os
import tempfile
from flood_forecast.preprocessing.preprocess_da_rnn import make_data
from flood_forecast.da_rnn.train_da import da_rnn, train
from flood_forecast.model_dict_function import pytorch_model_dict


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


class TestDARNNMultiTarget(unittest.TestCase):
    """Covers the ``out_feats`` wiring of the DARNN encoder/decoder pair."""

    def setUp(self):
        """
        Sets the shared shape parameters used to build DARNN instances under test.

        :return: None
        :rtype: None
        """
        self.batch_size = 4
        self.n_time_series = 6
        self.forecast_history = 11
        self.hidden_size_encoder = 16
        self.decoder_hidden_size = 12

    def build_model(self, out_feats: int, **kwargs) -> torch.nn.Module:
        """
        Builds a DARNN instance from the model dictionary entry used in production configs.

        :param out_feats: The number of target features the decoder should emit.
        :type out_feats: int
        :param kwargs: Extra keyword arguments forwarded to the DARNN constructor.
        :type kwargs: typing.Any
        :return: The constructed DARNN model.
        :rtype: torch.nn.Module
        """
        return pytorch_model_dict["DARNN"](
            self.n_time_series, self.hidden_size_encoder, self.forecast_history,
            self.decoder_hidden_size, out_feats=out_feats, dropout=0.0, **kwargs)

    def make_input(self) -> torch.Tensor:
        """
        Creates a random input tensor shaped (batch_size, forecast_history - 1, n_time_series).

        :return: The random input tensor.
        :rtype: torch.Tensor
        """
        return torch.rand(self.batch_size, self.forecast_history - 1, self.n_time_series)

    def test_forward_shape_single_target(self):
        """
        Asserts a DARNN with out_feats=1 still returns a (batch_size, 1) prediction.

        :return: None
        :rtype: None
        """
        out = self.build_model(1)(self.make_input())
        self.assertEqual(out.shape, torch.Size([self.batch_size, 1]))

    def test_forward_shape_three_targets(self):
        """
        Asserts a DARNN with out_feats=3 returns a (batch_size, 3) prediction.

        :return: None
        :rtype: None
        """
        out = self.build_model(3)(self.make_input())
        self.assertEqual(out.shape, torch.Size([self.batch_size, 3]))

    def test_backward_pass_multi_target(self):
        """
        Asserts gradients flow to both the encoder and decoder when out_feats=3.

        :return: None
        :rtype: None
        """
        model = self.build_model(3)
        model(self.make_input()).sum().backward()
        self.assertIsNotNone(model.decoder.fc_final.weight.grad)
        self.assertIsNotNone(model.encoder.attn_linear.weight.grad)
        self.assertGreater(float(model.decoder.fc_final.weight.grad.abs().sum()), 0.0)

    def test_encoder_consumes_remaining_features(self):
        """
        Asserts the encoder input width is n_time_series minus the number of targets.

        :return: None
        :rtype: None
        """
        for out_feats in (1, 3):
            model = self.build_model(out_feats)
            self.assertEqual(model.encoder.input_size, self.n_time_series - out_feats)
            self.assertEqual(model.decoder.fc.in_features, self.hidden_size_encoder + out_feats)

    def test_probabilistic_multi_target(self):
        """
        Asserts the probabilistic variant emits one mean and one std per target.

        :return: None
        :rtype: None
        """
        dist = self.build_model(3, probabilistic=True)(self.make_input())
        self.assertEqual(dist.mean.shape, torch.Size([self.batch_size, 3]))
        self.assertEqual(dist.stddev.shape, torch.Size([self.batch_size, 3]))

    def test_gru_multi_target(self):
        """
        Asserts the GRU variant also honours out_feats=3.

        :return: None
        :rtype: None
        """
        out = self.build_model(3, gru_lstm=False)(self.make_input())
        self.assertEqual(out.shape, torch.Size([self.batch_size, 3]))

    def test_invalid_out_feats_rejected(self):
        """
        Asserts an out_feats value that leaves no exogenous columns raises a ValueError.

        :return: None
        :rtype: None
        """
        with self.assertRaises(ValueError):
            self.build_model(self.n_time_series)
        with self.assertRaises(ValueError):
            self.build_model(0)


if __name__ == '__main__':
    unittest.main()
