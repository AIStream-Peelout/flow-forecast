import torch
import numpy
import unittest
from flood_forecast.meta_models.basic_ae import AE
from flood_forecast.utils import numpy_to_tvar


class MetaModels(unittest.TestCase):
    def setUp(self):
        self.AE = AE(10, 128)

    def test_ae_init(self):
        self.assertEqual(self.AE.encoder_hidden_layer.in_features, 10)
        self.assertEqual(self.AE(torch.rand(2, 10)).shape[0], 2)

    def test_ae_2(self):
        self.assertEqual(self.AE.decoder_output_layer.out_features, 10)
        res = numpy_to_tvar(numpy.random.rand(1, 2))
        self.assertIsInstance(res, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
