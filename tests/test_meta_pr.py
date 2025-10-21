import torch
import numpy
import unittest
from flood_forecast.meta_models.basic_ae import AE
from flood_forecast.utils import numpy_to_tvar


class MetaModels(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test environment by initializing an AutoEncoder (AE) model with
        an input dimension of 10 and a hidden dimension of 128.

        :return: None
        :rtype: None
        """

        self.AE = AE(10, 128)

    def test_ae_init(self) -> None:
        """
        Test that the AE model is initialized correctly by verifying the input feature size
        of the encoder and checking the output shape after a forward pass.

        :return: None
        :rtype: None
        """

        self.assertEqual(self.AE.encoder_hidden_layer.in_features, 10)
        self.assertEqual(self.AE(torch.rand(2, 10)).shape[0], 2)

    def test_ae_2(self) -> None:
        """
        Validate the output feature size of the AE model’s decoder and confirm that the
        utility function `numpy_to_tvar` correctly converts a NumPy array to a PyTorch tensor.

        :return: None
        :rtype: None
        """

        self.assertEqual(self.AE.decoder_output_layer.out_features, 10)
        res = numpy_to_tvar(numpy.random.rand(1, 2))
        self.assertIsInstance(res, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
