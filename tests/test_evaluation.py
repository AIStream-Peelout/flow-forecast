import os
import unittest
import datetime
from sklearn.preprocessing import StandardScaler
from flood_forecast.time_model import PyTorchForecast
from flood_forecast.evaluator import infer_on_torch_model, evaluate_model
import torch
import numpy
# Set random seed for same.
numpy.random.seed(0)
torch.manual_seed(0)


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_init"
        )
        self.test_path2 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data"
        )
        self.model_params = {
            "model_name": "MultiAttnHeadSimple",
            "metrics": ["MSE", "MAPE"],
            "model_params": {"number_time_series": 3, "seq_len": 20},
            "dataset_params": {
                "forecast_history": 20,
                "class": "default",
                "forecast_length": 20,
                "relevant_cols": ["cfs", "temp", "precip"],
                "target_col": ["cfs"],
                "interpolate": False,
            },
            "wandb": False,
            "inference_params": {"hours_to_forecast": 15}
             }
        self.model_linear_params = {
            "model_name": "SimpleLinearModel",
            "metrics": ["MSE", "MAPE"],
            "use_decoder": True,
            "model_params": {
                "n_time_series": 3,
                "seq_length": 100,
                "output_seq_len": 20,
            },
            "dataset_params": {
                "forecast_history": 100,
                "class": "default",
                "forecast_length": 20,
                "relevant_cols": ["cfs", "temp", "precip"],
                "target_col": ["cfs"],
                "interpolate": False,
                "train_end": 50,
                "valid_end": 100,
            },
            "training_params": {
                "optimizer": "Adam",
                "lr": 0.01,
                "criterion": "MSE",
                "epochs": 1,
                "batch_size": 2,
                "optim_params": {},
            },
            "wandb": False,
            "inference_params": {"hours_to_forecast": 15},
        }
        keag_file = os.path.join(self.test_path, "keag_small.csv")
        self.model = PyTorchForecast(
            "MultiAttnHeadSimple", keag_file, keag_file, keag_file, self.model_params
        )
        self.linear_model = PyTorchForecast(
            "SimpleLinearModel",
            keag_file,
            keag_file,
            keag_file,
            self.model_linear_params,
        )
        self.data_base_params = {
            "file_path": os.path.join(self.test_path2, "keag_small.csv"),
            "forecast_history": 20,
            "forecast_length": 20,
            "relevant_cols": ["cfs", "temp", "precip"],
            "target_col": ["cfs"],
            "interpolate_param": False,
        }
        self.data_base_params_with_scaling = {
            "file_path": os.path.join(self.test_path2, "keag_small.csv"),
            "forecast_history": 20,
            "forecast_length": 20,
            "relevant_cols": ["cfs", "temp", "precip"],
            "target_col": ["cfs"],
            "interpolate_param": False,
            "scaling": StandardScaler(),
        }

    def test_infer_on_torch(self):
        df, end_tensor, hist, idx, test_data, prediction_samples = infer_on_torch_model(
            self.model,
            os.path.join(self.test_path2, "keag_small.csv"),
            datetime_start=datetime.datetime(2014, 6, 2, 0),
            dataset_params=self.data_base_params,
        )
        self.assertEqual(end_tensor.shape[0], 336)
        self.assertEqual(df.iloc[0]["preds"], 0)
        self.assertNotEqual(df.iloc[22]["preds"], 0)
        self.assertEqual(idx, 759)

    def test_evaluator(self):
        inference_params = {
            "datetime_start": datetime.datetime(2016, 5, 31, 0),
            "hours_to_forecast": 336,
            "dataset_params": self.data_base_params,
            "test_csv_path": os.path.join(self.test_path2, "keag_small.csv"),
        }
        model_result = evaluate_model(
            self.model, "PyTorch", ["cfs"], ["MSE", "L1"], inference_params, {}
        )
        print(model_result)
        eval_dict = model_result[0]
        self.assertGreater(eval_dict["cfs_MAPELoss"], 0)
        self.assertGreater(eval_dict["cfs_MSELoss"], 420)
        self.assertNotAlmostEqual(eval_dict["cfs_MAPELoss"].item(), eval_dict["cfs_MSELoss"].item())
        # self.assertLessEqual(eval_dict["cfs_MAPELoss"].item(), 400)

    @unittest.skip("Issues with the prediction samples param")
    def test_evaluator_generate_prediction_samples(self):
        inference_params = {
            "datetime_start": datetime.datetime(2016, 5, 31, 0),
            "hours_to_forecast": 336,
            "dataset_params": self.data_base_params,
            "test_csv_path": os.path.join(self.test_path2, "keag_small.csv"),
            "num_prediction_samples": 100,
        }
        model_result = evaluate_model(
            self.model, "PyTorch", ["cfs"], ["MSE", "L1"], inference_params, {}
        )
        df_train_and_test = model_result[1]
        df_prediction_samples = model_result[3]
        self.assertTrue(df_train_and_test.index.equals(df_prediction_samples[0].index))
        self.assertEqual(100, df_prediction_samples[0].shape[1])

    def test_evaluator_with_scaling_not_equal_without_scaling(self):
        inference_params = {
            "datetime_start": datetime.datetime(2016, 5, 31, 0),
            "hours_to_forecast": 336,
            "dataset_params": self.data_base_params,
            "test_csv_path": os.path.join(self.test_path2, "keag_small.csv"),
        }
        inference_params_with_scaling = {
            "datetime_start": datetime.datetime(2016, 5, 31, 0),
            "hours_to_forecast": 336,
            "dataset_params": self.data_base_params_with_scaling,
            "test_csv_path": os.path.join(self.test_path2, "keag_small.csv"),
        }
        model_result_1 = evaluate_model(
            self.model, "PyTorch", ["cfs"], ["MSE", "L1"], inference_params, {}
        )
        model_result_2 = evaluate_model(
            self.model,
            "PyTorch",
            ["cfs"],
            ["MSE", "L1"],
            inference_params_with_scaling,
            {},
        )
        self.assertFalse(model_result_1[1]["preds"].equals(model_result_2[1]["preds"]))

    @unittest.skip("Issues with the prediction samples param")
    def test_evaluator_df_preds_with_scaling_not_equal_without_scaling(self):
        inference_params = {
            "datetime_start": datetime.datetime(2016, 5, 31, 0),
            "hours_to_forecast": 336,
            "dataset_params": self.data_base_params,
            "test_csv_path": os.path.join(self.test_path2, "keag_small.csv"),
            "num_prediction_samples": 10,
        }
        inference_params_with_scaling = {
            "datetime_start": datetime.datetime(2016, 5, 31, 0),
            "hours_to_forecast": 336,
            "dataset_params": self.data_base_params_with_scaling,
            "test_csv_path": os.path.join(self.test_path2, "keag_small.csv"),
            "num_prediction_samples": 10,
        }
        model_result_1 = evaluate_model(
            self.model, "PyTorch", ["cfs"], ["MSE", "L1"], inference_params, {}
        )
        model_result_2 = evaluate_model(
            self.model,
            "PyTorch",
            ["cfs"],
            ["MSE", "L1"],
            inference_params_with_scaling,
            {},
        )
        print(model_result_1)
        self.assertFalse(model_result_1[3][0].equals(model_result_2[3][0]))

    def test_linear_decoder(self):
        decoder_params = {"decoder_function": "simple_decode", "unsqueeze_dim": 1}
        self.data_base_params["forecast_history"] = 100
        inference_params = {
            "datetime_start": datetime.datetime(2016, 5, 31, 0),
            "hours_to_forecast": 336,
            "dataset_params": self.data_base_params,
            "test_csv_path": os.path.join(self.test_path2, "keag_small.csv"),
            "decoder_params": decoder_params,
        }
        infer_on_torch_model(self.linear_model, **inference_params)

    def test_outputs_different(self):
        pass

if __name__ == "__main__":
    unittest.main()
