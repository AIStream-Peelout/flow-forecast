import unittest
import os
import torch
from torch.utils.data import DataLoader
from flood_forecast.preprocessing.pytorch_loaders import CSVDataLoader
from flood_forecast.transformer_xl.transformer_basic import SimpleTransformer, greedy_decode, generate_square_subsequent_mask

class TestDecoding(unittest.TestCase):
    def setUp(self):
        self.model = SimpleTransformer(3, 48)
        self.data_test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_init", "chick_final.csv")
        self.validation_loader = DataLoader(CSVDataLoader(self.data_test_path, forecast_history=30, forecast_length=20, target_col=['cfs'], relevant_cols=['cfs','temp','precip']), shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
        self.sequence_size = 48
    
    def test_full_forward_method(self):
        test_data = torch.rand(1, 48, 3)
        result = self.model(test_data)
        self.assertEqual(result.shape, torch.Size([1, 48]))
    
    def test_encoder_seq(self):
        test_data = torch.rand(1, 3, 48)
        result = self.model.encode_sequence(test_data)
        self.assertEqual(result.shape, torch.Size([1, 48, 128]))

    def test_for_leakage(self):
        """
        Simple test to check that raw target data does NOT
        leak during validation.
        """
        src_mask = generate_square_subsequent_mask(self.sequence_size)
        src, trg = next(iter(self.validation_loader))
        trg_mem = trg.clone().detach()
        result = greedy_decode(self.model, src, src_mask, self.sequence_size, trg, src)
        self.assertNotEqual(result[0, 2, 0], trg_mem[0, 1, 0])
        self.assertEqual(result[0, 2, 1], trg_mem[0, 1, 1])
        self.assertEqual(result[0, 2, 2], trg_mem[0, 1, 2])
        
    def test_make_function(self):
        self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main()