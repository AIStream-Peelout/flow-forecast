import unittest
from flood_forecast.transformer_xl.informer import Informer
from flood_forecast.transformer_xl.data_embedding import DataEmbedding
from flood_forecast.preprocessing.pytorch_loaders import TemporalLoader, TemporalTestLoader
from flood_forecast.temporal_decoding import decoding_function
import torch


class TestInformer(unittest.TestCase):
    def setUp(self):
        self.informer = Informer(3, 3, 3, 20, 20, 20, factor=1)
        self.kwargs = {
                    "file_path": "tests/test_data/keag_small.csv",
                    "forecast_history": 5,
                    "forecast_length": 1,
                    "target_col": ["cfs"],
                    "relevant_cols": ["cfs", "temp", "precip"],
                    "sort_column": "date",
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
        loa = TemporalLoader(["month", "day", "day_of_week", "hour"], self.kwargs)
        result = loa[0]
        self.assertEqual(len(result), 2)
        # Test output has proper dimensions
        # print(loa[0][0].shape)
        self.assertEqual(result[0][0].shape[0], 5)
        self.assertEqual(result[0][1].shape[1], 4)
        self.assertEqual(result[0][0].shape[1], 3)
        self.assertEqual(result[0][1].shape[0], 5)
        # Test output right order
        temporal_src_embd = result[0][1]
        second = temporal_src_embd[2, :]
        self.assertEqual(second[0], 5)
        self.assertEqual(second[1], 1)
        self.assertEqual(second[3], 3)
        # Test data loading component
        d = DataEmbedding(3, 128)
        embedding = d(result[0][0].unsqueeze(0), temporal_src_embd.unsqueeze(0))
        self.assertEqual(embedding.shape[2], 128)
        i = Informer(3, 3, 3, 5, 5, out_len=4, factor=1)
        r0 = result[0][0].unsqueeze(0)
        r1 = result[0][1].unsqueeze(0)
        r3 = result[1][1].unsqueeze(0)
        r2 = result[1][0].unsqueeze(0)
        res = i(r0, r1, r3, r2)
        self.assertEqual(res.shape[1], 1)
        self.assertEqual(r3[0, 0, 0].item(), 459)

    def test_temporal_load(self):
        loa = TemporalLoader(["month", "day", "day_of_week", "hour"], self.kwargs, 2)
        data = loa[0]
        self.assertEqual(data[1][1].shape[0], 3)
        self.assertEqual(data[1][1][0, 0].item(), 449)
        self.assertEqual(data[1][1][2, 0], 459)

    def test_data_temporal_loader_init(self):
        kwargs2 = self.kwargs.copy()
        kwargs3 = {
            "forecast_total": 336,
            "df_path": "tests/test_data2/keag_small.csv",
            "kwargs": kwargs2
        }
        d = TemporalTestLoader(["month", "day", "day_of_week", "hour"], kwargs3, 3)
        src, trg, df, _ = d[0]
        self.assertEqual(trg[0].shape[0], 339)
        self.assertEqual(src[0].shape[0], 5)
        self.assertEqual(len(df.index), 341)
        self.assertEqual(trg[1][0, 0].item(), 445)

    def test_decodign_t(self):
        src = torch.rand(20, 3)
        trg = torch.rand(355, 3)
        src1 = torch.rand(20, 4)
        trg1 = torch.rand(355, 4)
        d = decoding_function(self.informer, src, trg, 5, src1, trg1, 1, 20, 336, "cpu")
        self.assertEqual(d.shape[0], 1)
        self.assertEqual(d.shape[1], 336)

    def test_decoding_2(self):
        src = torch.rand(20, 3)
        trg = torch.rand(355, 3)
        src1 = torch.rand(20, 4)
        trg1 = torch.rand(355, 4)
        d = decoding_function(self.informer, src, trg, 5, src1, trg1, 1, 20, 336, "cpu")
        self.assertEqual(d.shape[0], 1)
        self.assertEqual(d.shape[1], 336)
        self.assertNotEqual(d[0, 0, 0].item(), d[0, 1, 0].item())
        self.assertNotAlmostEqual(d[0, 0, 0].item(), d[0, 330, 0].item())
        self.assertNotAlmostEqual(d[0, 20, 0].item(), d[0, 333, 0].item())
        self.assertNotAlmostEqual(d[0, 300, 0].item(), d[0, 334, 0].item())
        self.assertNotAlmostEqual(d[0, 20, 0].item(), trg[20, 0].item())
        self.assertNotAlmostEqual(d[0, 21, 0].item(), trg[21, 0].item())

    def test_decoding_3(self):
        informer_model2 = Informer(3, 3, 3, 48, 24, 12, factor=1)
        src = torch.rand(1, 48, 3)
        trg = torch.rand(1, 362, 3)
        src1 = torch.rand(1, 48, 4)
        trg1 = torch.rand(1, 362, 4)
        d = decoding_function(informer_model2, src, trg, 12, src1, trg1, 1, 36, 336, "cpu")
        self.assertEqual(d.shape[0], 1)
        self.assertEqual(d.shape[1], 336)

    def test_t_loade2(self):
        s_wargs = {
                    "file_path": "tests/test_data/keag_small.csv",
                    "forecast_history": 39,
                    "forecast_length": 2,
                    "target_col": ["cfs"],
                    "relevant_cols": ["cfs", "temp", "precip"],
                    "sort_column": "date",
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
        s_wargs["forecast_history"] = 39
        t_load = TemporalLoader(["month", "day"], s_wargs, 30)
        src, trg = t_load[0]
        self.assertEqual(trg[1].shape[0], 32)
        self.assertEqual(trg[0].shape[0], 32)
        self.assertEqual(trg[1].shape[1], 5)
        self.assertEqual(trg[0].shape[1], 2)
        #  this test makes sure the label_lens parameter works
        print("Complete")
