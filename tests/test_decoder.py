import unittest
from flood_forecast.transformer_xl.transformer_basic import SimpleTransformer, greedy_decode

class TestDecoding(unittest.TestCase):
    def setUp(self):
        self.model = SimpleTransformer(3, 48)
        self.validation_loader = validation_data_loader = DataLoader(validation_data, batch_size=8, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
        self.sequence_size = 48

    def test_for_leakage(self):
        src_mask = generate_square_subsequent_mask(sequence_size)
        src, trg = next(iter(validation_data_loader))
        trg_mem = trg.clone().detach()
        print(trg_mem[0,1,0])
        result = greedy_decode(self.model, src, src_mask, self.sequence_size, trg, src)
        print(trg_mem[0,1,0])
        self.assertNotEqual(result[0,2,0], trg_mem[0,1,0])
        self.assertEqual(result[0,2,1], trg_mem[0, 1, 1])
        self.assertEqual(result[0,2,2], trg_mem[0, 1, 2])
        
    def test_make_function(self):
        self.assertEqual(1,1)