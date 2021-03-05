import unittest
from flood_forecast.transformer_xl.informer import Informer
from flood_forecast.transformer_xl.data_embedding import DataEmbedding
from flood_forecast.preprocessing.pytorch_loaders import TemporalLoader
import torch


class TestInformer(unittest.TestCase):
    def setUp(self):
        self.informer = Informer(3, 3, 3, 20, 20, 20, factor=1)

    def test_informer(self):
        # Format should be (batch_size, seq_len, n_time_series) (batch_size, seq_len,,)
        result = self.informer(torch.rand(2, 20, 3), torch.rand(2, 20, 4), torch.rand(2, 20, 3), torch.rand(2, 20, 4))
        self.assertEqual(len(result.shape), 3)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 20)

    def test_data_embedding(self):
        d = DataEmbedding(5, 128, data=5)
        r = d(torch.rand(2, 10, 5), torch.rand(2, 10, 5))
        self.assertTrue(hasattr(d.temporal_embedding, "month_embed"))
        self.assertEqual(r.shape[2], 128)

    def test_temporal_loader(self):
        kwargs = {
                    "file_path": "tests/test_data/keag_small.csv",
                    "forecast_history": 5,
                    "forecast_length": 1,
                    "feature_params":
                    {
                        "datetime_params": {
                            "month": "numerical",
                            "day": "numerical",
                            "day_of_week": "numerical",
                            "hour": "numerical"
                        }
                    }
                }
        loa = TemporalLoader(["month", "day", "weekday", "hour"], kwargs)
        result = loa.__getitem__(0)
        self.assertEqual(len(result), 4)

    #  def test_different_minute(self): e
    #  d = DataEmbedding(5, 128, data=5)
    #  r = d(torch.rand(5, 128, 5))
    #  d1 = DataEmbedding(5, 128, data=4)
    #  r1= d(torch.rand)
    #  self.assertNotAlmostEqual(r, r)
