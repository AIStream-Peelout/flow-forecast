from flood_forecast.meta_models.merging_model import MergingModel, MultiModalSelfAttention
from flood_forecast.utils import make_criterion_functions
import unittest
import torch


class TestMerging(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test environment by initializing different MergingModel instances
        and a MultiModalSelfAttention module to be used in multiple test cases.

        :return: None
        :rtype: None
        """

        self.merging_model = MergingModel("Concat", {"cat_dim": 2, "repeat": True})
        self.merging_model_bi = MergingModel("Bilinear", {"in1_features": 6, "in2_features": 3 - 2, "out_features": 40})
        self.merging_model_2 = MergingModel("Bilinear2", {"in1_features": 20, "in2_features": 25, "out_features": 49})
        self.merging_mode3 = MergingModel("Concat", {"cat_dim": 2, "repeat": True, "use_layer": True, "out_shape": 10,
                                                     "combined_d": 15})
        self.attn = MultiModalSelfAttention(128, 4, 0.2)

    def test_merger_runs(self) -> None:
        """
        Test that the concatenation-based MergingModel produces an output with expected shape
        when given inputs of specific dimensions.

        :return: None
        :rtype: None
        """

        m = self.merging_model(torch.rand(2, 6, 10), torch.rand(4))
        self.assertEqual(m.shape[0], 2)
        self.assertEqual(m.shape[1], 6)
        self.assertEqual(m.shape[2], 14)

    def test_merger_two(self) -> None:
        """
        Test that the concatenation-based MergingModel correctly adjusts output width
        when a different input sequence length is provided.

        :return: None
        :rtype: None
        """

        m = self.merging_model(torch.rand(2, 6, 20), torch.rand(4))
        self.assertEqual(m.shape[2], 24)

    def test_crit_functions_list(self) -> None:
        """
        Verify that make_criterion_functions returns a list when provided with
        a list of loss function names.

        :return: None
        :rtype: None
        """
        res = make_criterion_functions(["MSE", "RMSE", "MAPE"])
        self.assertIsInstance(res, list)

    def test_crit_functions_dict(self) -> None:
        """
        Verify that make_criterion_functions returns a list when provided with
        a dictionary of loss names and their parameters.

        :return: None
        :rtype: None
        """

        res = make_criterion_functions({"MASELoss": {"baseline_method": "mean"}, "MSE": {}})
        self.assertIsInstance(res, list)

    def test_bilinear_model(self) -> None:
        """
        Validate that the Bilinear merging model returns a tensor with the correct
        output dimension for the second axis.

        :return: None
        :rtype: None
        """

        r = self.merging_model_bi(torch.rand(2, 6, 128), torch.rand(128))
        self.assertEqual(r.shape[1], 40)

    def test_bilinear_2(self) -> None:
        """
        Check the Bilinear2 merging model produces the expected output shape
        when merging two input tensors of specified shape.

        :return: None
        :rtype: None
        """

        m = self.merging_model_2(torch.rand(2, 6, 20), torch.rand(25))
        self.assertEqual(m.shape[2], 49)

    def test_cat_out(self) -> None:
        """
        Ensure the Concat merging model using a layer produces the correct output shape.

        :return: None
        :rtype: None
        """

        m = self.merging_mode3(torch.rand(2, 6, 10), torch.rand(5))
        self.assertEqual(m.shape[2], 10)

if __name__ == '__main__':
    unittest.main()
