import unittest
import torch
from flood_forecast.basic.narx import NARX


class TestNARX(unittest.TestCase):
    def setUp(self):
        """
        Creates a default NARX model and a sample input batch for testing.

        :return: None
        :rtype: None
        """
        self.batch_size = 4
        self.forecast_history = 10
        self.n_time_series = 3
        self.narx = NARX(n_time_series=self.n_time_series, forecast_history=self.forecast_history)
        self.sample_batch = torch.rand(self.batch_size, self.forecast_history, self.n_time_series)

    def test_narx_forward_shape(self):
        """
        Tests that the forward pass returns the correct output shape for the default single-step forecast.

        :return: None
        :rtype: None
        """
        result = self.narx(self.sample_batch)
        self.assertEqual(result.shape, (self.batch_size, 1))

    def test_narx_multi_step_output(self):
        """
        Tests that the forward pass returns the correct shape when output_seq_len is greater than one.

        :return: None
        :rtype: None
        """
        narx = NARX(n_time_series=self.n_time_series, forecast_history=self.forecast_history, output_seq_len=5)
        result = narx(self.sample_batch)
        self.assertEqual(result.shape, (self.batch_size, 5))

    def test_narx_partial_lags(self):
        """
        Tests that the model works with target and exogenous lag orders smaller than the forecast history.

        :return: None
        :rtype: None
        """
        narx = NARX(n_time_series=self.n_time_series, forecast_history=self.forecast_history,
                    n_target_lags=3, n_exog_lags=6, num_hidden_layers=2, dropout=0.1)
        result = narx(self.sample_batch)
        self.assertEqual(result.shape, (self.batch_size, 1))

    def test_narx_no_exogenous(self):
        """
        Tests that the model handles the purely autoregressive case where all series are targets.

        :return: None
        :rtype: None
        """
        narx = NARX(n_time_series=1, forecast_history=self.forecast_history, n_targets=1)
        result = narx(torch.rand(self.batch_size, self.forecast_history, 1))
        self.assertEqual(result.shape, (self.batch_size, 1))

    def test_narx_probabilistic(self):
        """
        Tests that the probabilistic mode returns a Normal distribution.

        :return: None
        :rtype: None
        """
        narx = NARX(n_time_series=self.n_time_series, forecast_history=self.forecast_history, probabilistic=True)
        result = narx(self.sample_batch)
        self.assertIsInstance(result, torch.distributions.Normal)
        self.assertEqual(result.mean.shape, (self.batch_size, 1))

    def test_narx_gradient_flow(self):
        """
        Tests that gradients propagate back to the first MLP layer after a backward pass.

        :return: None
        :rtype: None
        """
        result = self.narx(self.sample_batch)
        result.sum().backward()
        self.assertIsNotNone(self.narx.mlp[0].weight.grad)

    def test_narx_invalid_params(self):
        """
        Tests that invalid lag orders, target counts, and activations raise a ValueError.

        :return: None
        :rtype: None
        """
        with self.assertRaises(ValueError):
            NARX(n_time_series=3, forecast_history=10, n_target_lags=11)
        with self.assertRaises(ValueError):
            NARX(n_time_series=3, forecast_history=10, n_exog_lags=12)
        with self.assertRaises(ValueError):
            NARX(n_time_series=3, forecast_history=10, n_targets=4)
        with self.assertRaises(ValueError):
            NARX(n_time_series=3, forecast_history=10, activation="gelu")


if __name__ == '__main__':
    unittest.main()
