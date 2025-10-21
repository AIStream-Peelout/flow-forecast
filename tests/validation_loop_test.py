import unittest
import torch
from flood_forecast.basic.linear_regression import SimpleLinearModel, handle_gaussian_loss
from flood_forecast.meta_models.basic_ae import AE
from flood_forecast.basic.base_line_methods import NaiveBase
from flood_forecast.custom.custom_activation import _roll_last


class TestBasicMethodVal(unittest.TestCase):

    def test_simple_linear_prob(self):
        """
        Test that SimpleLinearModel returns a torch Normal distribution
        when passed a random tensor input.

        :return: None
        :rtype: None
        """
        s = SimpleLinearModel(9, 3, 1, True)
        r = s(torch.rand(4, 9, 3))
        self.assertIsInstance(r, torch.distributions.Normal)

    def test_handle_gaussian_loss(self):
        """
        Test the handle_gaussian_loss function with a tuple of random tensors
        and print the result.

        :return: None
        :rtype: None
        """
        result = handle_gaussian_loss((torch.rand(10, 2), torch.rand(10, 2)))
        print(result)

    def test_hano_scaling(self):
        """
        Test NaiveBase model output shape given a random tensor input.

        :return: None
        :rtype: None
        """
        n = NaiveBase(20, 10, 1)
        e = n(torch.rand(4, 20, 10))
        self.assertEqual(e.shape[1], 1)

    def test_ae(self):
        """
        Test AE model's generate_representation method output shape
        given a random tensor input.

        :return: None
        :rtype: None
        """
        ae = AE(9, 128)
        rep = ae.generate_representation(torch.rand(4, 9))
        self.assertEqual(rep.shape, (4, 128))

    def new_test(self):
        _roll_last(torch.rand(43, 4), 1)

if __name__ == '__main__':
    unittest.main()
