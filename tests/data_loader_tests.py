from flood_forecast.preprocessing.pytorch_loaders import (
    CSVTestLoader,
    CSVDataLoader,
    AEDataloader,
)
import unittest
import os
import torch
from datetime import datetime


class DataLoaderTests(unittest.TestCase):
    """
    Class to test data loader functionality for the code mod.
    Specifically, reuturn types and indexing to make sure there is no overlap.
    """

    def setUp(self):
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data"
        )
        data_base_params = {
            "file_path": os.path.join(self.test_data_path, "keag_small.csv"),
            "forecast_history": 20,
            "forecast_length": 20,
            "relevant_cols": ["cfs", "temp", "precip"],
            "target_col": ["cfs"],
            "interpolate_param": False,
        }
        self.train_loader = CSVDataLoader(
            os.path.join(self.test_data_path, "keag_small.csv"),
            30,
            20,
            target_col=["cfs"],
            relevant_cols=["cfs", "precip", "temp"],
            interpolate_param=False,
        )
        data_base_params["start_stamp"] = 20
        self.test_loader = CSVTestLoader(
            os.path.join(self.test_data_path, "keag_small.csv"),
            336,
            **data_base_params
        )
        self.ae_loader = AEDataloader(
            os.path.join(self.test_data_path, "keag_small.csv"),
            relevant_cols=["cfs", "temp", "precip"],
        )
        data_base_params["end_stamp"] = 220
        self.train_loader2 = CSVDataLoader(
            **data_base_params
        )

    def test_loader2_get_item(self):
        src, df, forecast_start_index = self.test_loader[0]
        self.assertEqual(type(src), torch.Tensor)
        self.assertEqual(forecast_start_index, 20)
        self.assertEqual(df.iloc[2]["cfs"], 445)
        self.assertEqual(len(df), 356)

    def test_loader2_get_date(self):
        src, df, forecast_start_index, = self.test_loader.get_from_start_date(
            datetime(2014, 6, 3, 0)
        )
        self.assertEqual(type(src), torch.Tensor)
        self.assertEqual(forecast_start_index, 783)
        self.assertEqual(
            df.iloc[0]["datetime"].day, datetime(2014, 6, 2, 4).day
        )

    def test_loader_get_gcs_data(self):
        test_loader = CSVDataLoader(
            file_path="gs://flow_datasets/Afghanistan____.csv",
            forecast_history=14,
            forecast_length=14,
            target_col=["cases"],
            relevant_cols=["cases", "recovered", "active", "deaths"],
            sort_column="date",
            interpolate_param=False,
            gcp_service_key=None,  # for CircleCI tests, local test needs key.json
        )
        self.assertIsInstance(test_loader, CSVDataLoader)

    def test_ae(self):
        x, y = self.ae_loader[0]
        self.assertEqual(x.shape, y.squeeze(1).shape)

    def test_trainer(self):
        x, y = self.train_loader[0]
        self.assertEqual(x.shape[0], 30)
        self.assertEqual(x.shape[1], 3)
        self.assertEqual(y.shape[0], 20)
        # Check first and last dim are not overlap
        self.assertFalse(torch.eq(x[29, 0], y[0, 0]))

    def test_start_end(self):
        self.assertEqual(len(self.train_loader.df), len(self.test_loader.df) + 20)
        self.assertEqual(len(self.train_loader2.df), 200)


if __name__ == "__main__":
    unittest.main()
