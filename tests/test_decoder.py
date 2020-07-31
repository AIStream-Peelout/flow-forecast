import unittest
import os
import torch
from torch.utils.data import DataLoader
from flood_forecast.preprocessing.pytorch_loaders import CSVDataLoader
from flood_forecast.model_dict_function import pytorch_criterion_dict
from flood_forecast.transformer_xl.transformer_basic import SimpleTransformer, greedy_decode


class TestDecoding(unittest.TestCase):
    def setUp(self):
        self.model = SimpleTransformer(3, 30, 20)
        self.data_test_path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "test_init",
            "chick_final.csv")
        self.validation_loader = DataLoader(
            CSVDataLoader(
                self.data_test_path,
                forecast_history=30,
                forecast_length=20,
                target_col=['cfs'],
                relevant_cols=[
                    'cfs',
                    'temp',
                    'precip'],
                interpolate_param=False),
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None)
        self.sequence_size = 30

    def test_full_forward_method(self):
        test_data = torch.rand(1, 30, 3)
        result = self.model(test_data, t=torch.rand(1, 20, 3))
        self.assertEqual(result.shape, torch.Size([1, 20]))

    def test_encoder_seq(self):
        test_data = torch.rand(1, 30, 3)
        result = self.model.encode_sequence(test_data)
        self.assertEqual(result.shape, torch.Size([30, 1, 128]))

    def test_for_leakage(self):
        """
        Simple test to check that raw target data does NOT
        leak during validation steps.
        """
        src, trg = next(iter(self.validation_loader))
        trg_mem = trg.clone().detach()
        result = greedy_decode(self.model, src, 20, trg)
        self.assertNotEqual(result[0, 1, 0], trg_mem[0, 1, 0])
        self.assertEqual(result[0, 1, 1], trg_mem[0, 1, 1])
        self.assertEqual(result[0, 1, 2], trg_mem[0, 1, 2])
        loss = pytorch_criterion_dict["MSE"]()(trg, trg_mem)

        self.assertNotEqual(result[0, 1, 0], result[0, 4, 0])
        self.assertGreater(loss, 0)

    def test_make_function(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()
