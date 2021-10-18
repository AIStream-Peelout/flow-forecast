import unittest
import torch
from flood_forecast.basic.linear_regression import SimpleLinearModel, handle_gaussian_loss
from flood_forecast.meta_models.basic_ae import AE
from flood_forecast.basic.base_line_methods import NaiveBase
from flood_forecast.custom.custom_activation import _roll_last


class TestBasicMethodVal(unittest.TestCase):

    def test_simple_linear_prob(self):
        s = SimpleLinearModel(9, 3, 1, True)
        r = s(torch.rand(4, 9, 3))
        self.assertIsInstance(r, torch.distributions.Normal)

    def test_handle_gaussian_loss(self):
        result = handle_gaussian_loss((torch.rand(10, 2), torch.rand(10, 2)))
        print(result)

    def test_hano_scaling(self):
        n = NaiveBase(20, 10, 1)
        e = n(torch.rand(4, 20, 10))
        self.assertEqual(e.shape, (4, 1))

    def test_ae(self):
        ae = AE(9, 128)
        rep = ae.generate_representation(torch.rand(4, 9))
        self.assertEqual(rep.shape, (4, 128))

    def new_test(self):
        _roll_last(torch.rand(43, 4), 1)

if __name__ == '__main__':
    unittest.main()