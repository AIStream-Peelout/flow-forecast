import unittest
import torch
from flood_forecast.basic.linear_regression import SimpleLinearModel, handle_gaussian_loss


class TestBasicMethodVal(unittest.TestCase):

    def test_simple_linear_prob(self):
        s = SimpleLinearModel(9, 3, 1, True)
        r = s(torch.rand(4, 9, 3))
        self.assertIsInstance(r, torch.nn.Distributions.Normal)

    def test_handle_gaussian_loss(self):
        handle_gaussian_loss((torch.rand(10, 2), torch.rand(10, 2)))

if __name__ == '__main__':
    unittest.main()
