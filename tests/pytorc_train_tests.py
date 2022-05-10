import os
import torch
from torch.utils.data import DataLoader
from flood_forecast.time_model import PyTorchForecast
from flood_forecast.custom.dilate_loss import DilateLoss
from flood_forecast.pytorch_training import torch_single_train, compute_loss
import unittest
import json
from flood_forecast.pytorch_training import train_transformer_style, handle_meta_data


class PyTorchTrainTests(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_init")
        self.model_params = {
            "metrics": ["MSE", "MAPE"],
            "model_params": {
                "number_time_series": 3,
                "seq_len": 20},
            "dataset_params": {
                "forecast_history": 20,
                "scaling": "StandardScaler",
                "class": "default",
                "forecast_length": 20,
                "relevant_cols": [
                    "cfs",
                    "temp",
                    "precip"],
                "target_col": ["cfs"],
                "interpolate": False},
            "training_params": {
                "optimizer": "Adam",
                "lr": .1,
                "criterion": "MSE",
                "epochs": 1,
                "batch_size": 2,
                "optim_params": {}},
            "wandb": False,
            "inference_params": {
                "hours_to_forecast": 10}}
        self.inf_params3 = {
            "metrics": ["MSE", "MAPE"],
            "use_decoder": True,
            "model_params": {
                "n_time_series": 3,
                "dec_in": 3,
                "c_out": 1,
                "seq_len": 20,
                "label_len": 10,
                "out_len": 2,
                "factor": 2},
            "dataset_params": {
                "forecast_history": 20,
                "scaling": "StandardScaler",
                "train_end": 2000,
                "valid_start": 900,
                "valid_end": 2025,
                "test_start": 900,
                "test_end": 2000,
                "class": "TemporalLoader",
                "temporal_feats": ["month", "day", "day_of_week", "hour"],
                "forecast_length": 2,
                "relevant_cols": [
                    "cfs",
                    "temp",
                    "precip"],
                "target_col": ["cfs"],
                "interpolate": False,
                "sort_column": "datetime",
                "feature_param":
                {
                    "datetime_params": {
                        "month": "numerical",
                        "day": "numerical",
                        "day_of_week": "numerical",
                        "hour": "numerical"
                    },
                }},
            "training_params": {
                "optimizer": "BertAdam",
                "criterion": "MSE",
                "epochs": 1,
                "batch_size": 100,
                "optim_params": {"lr": .1}},
            "wandb": False,
            "inference_params": {
                "hours_to_forecast": 11}}
        self.keag_file = os.path.join(self.test_path, "keag_small.csv")
        self.model = PyTorchForecast(
            "MultiAttnHeadSimple",
            self.keag_file,
            self.keag_file,
            self.keag_file,
            self.model_params)
        self.inf = PyTorchForecast("Informer", self.keag_file, self.keag_file,
                                   self.keag_file, self.inf_params3)
        self.dummy_model = PyTorchForecast(
            "DummyTorchModel", self.keag_file, self.keag_file, self.keag_file, {
                "model_params": {"forecast_length": 5},
                "metrics": ["MAPE", "MSE"],
                "dataset_params": {
                    "forecast_test_len": 15,
                    "num_workers": 2,
                    "forecast_history": 5,
                    "class": "default",
                    "forecast_length": 5,
                    "relevant_cols": ["cfs", "temp", "precip"],
                    "target_col": ["cfs"], "interpolate": False,
                    "train_end": 100
                },
                "training_params": {
                    "optimizer": "Adam",
                    "lr": .1,
                    "criterion": "MSE",
                    "epochs": 1,
                    "batch_size": 2,
                    "optim_params": {}
                },
                "inference_params": {"hours_to_forecast": 15},
                "wandb": False}
        )
        self.full_transformer_params = {
            "use_decoder": True,
            "model_params": {
                "number_time_series": 3,
                "seq_length": 20,
                "output_seq_len": 15},
            "metrics": ["MAPE", "MSE"],
            "dataset_params": {
                "forecast_history": 20,
                "class": "default",
                "forecast_length": 15,
                "relevant_cols": [
                    "cfs",
                    "temp",
                    "precip"],
                "target_col": ["cfs"],
                "interpolate": False,
                "train_end": 50,
                "valid_end": 100},
            "training_params": {
                "optimizer": "Adam",
                "lr": .01,
                "criterion": "MSE",
                "epochs": 1,
                "batch_size": 2,
                "optim_params": {}},
            "inference_params": {
                "hours_to_forecast": 100},
            "wandb": False}
        self.simple_param = {
            "use_decoder": True,
            "model_params": {
                "n_time_series": 3,
                "seq_length": 80,
                "output_seq_len": 20},
            "metrics": ["MAPE", "MSE"],
            "dataset_params": {
                "forecast_test_len": 25,
                "forecast_history": 20,
                "class": "default",
                "forecast_length": 15,
                "relevant_cols": [
                    "cfs",
                    "temp",
                    "precip"],
                "target_col": ["cfs"],
                "interpolate": False,
                "train_end": 50,
                "valid_end": 100},
            "inference_params": {
                "hours_to_forecast": 10},
            "training_params": {
                "optimizer": "Adam",
                "lr": .01,
                "criterion": "MSE",
                "epochs": 1,
                "batch_size": 2,
                "optim_params": {}},
            "wandb": False}
        self.transformer = PyTorchForecast(
            "SimpleTransformer",
            self.keag_file,
            self.keag_file,
            self.keag_file,
            self.full_transformer_params)
        self.simple_linear_model = PyTorchForecast(
            "SimpleLinearModel",
            self.keag_file,
            self.keag_file,
            self.keag_file,
            self.simple_param)
        self.opt = torch.optim.Adam(self.dummy_model.model.parameters(), lr=0.0001)
        self.criterion = torch.nn.modules.loss.MSELoss()
        self.data_loader = DataLoader(
            self.dummy_model.training,
            batch_size=2,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None)
        with open(os.path.join(os.path.dirname(__file__), "da_meta.json")) as f:
            self.meta_model_params = json.load(f)

    def test_pytorch_train_base(self):
        self.assertEqual(self.model.model.dense_shape.in_features, 3)
        self.assertEqual(self.model.model.multi_attn.embed_dim, 128)

    def test_train_and_resume(self):
        train_transformer_style(self.model, self.model_params["training_params"])
        self.assertEqual(len(os.listdir("model_save")), 2)
        print("first test passed")
        model2 = PyTorchForecast(
            "MultiAttnHeadSimple",
            self.keag_file,
            self.keag_file,
            self.keag_file,
            self.model_params)
        data = torch.rand(2, 20, 3)
        self.model_params["weight_path"] = os.path.join(
            "model_save", sorted(os.listdir("model_save"))[1])
        print("Moving to next test")
        model3 = PyTorchForecast(
            "MultiAttnHeadSimple",
            self.keag_file,
            self.keag_file,
            self.keag_file,
            self.model_params)
        print("model loaded")
        basic_model = model2.model
        basic_model.eval()
        print("more model stuff")
        pre_loaded_model = model3.model
        pre_loaded_model.eval()
        print("passed model stuff")
        self.assertFalse(torch.allclose(pre_loaded_model(data), basic_model(data)))
        print("first test good")
        self.assertTrue(torch.allclose(basic_model(data), basic_model(data)))

    def test_transfer_shit(self):
        self.model_params["weight_path"] = os.path.join(
            "model_save", sorted(os.listdir("model_save"))[1])
        self.model_params["model_params"]["output_seq_len"] = 6
        self.model_params["weight_path_add"] = {}
        model3 = PyTorchForecast(
            "MultiAttnHeadSimple",
            self.keag_file,
            self.keag_file,
            self.keag_file,
            self.model_params)
        # Assert shape is proper
        self.assertEqual(2, 2)
        data = torch.rand(1, 20, 3)
        self.assertEqual(model3.model(data).shape, torch.Size([1, 6]))

    def test_removing_layer_param(self):
        model3 = PyTorchForecast(
            "MultiAttnHeadSimple",
            self.keag_file,
            self.keag_file,
            self.keag_file,
            self.model_params)
        model3.save_model("output.pth", 2)
        self.model_params["model_params"]["output_seq_len"] = 7
        self.model_params["weight_path_add"] = {}
        self.model_params["weight_path_add"]["excluded_layers"] = [
            "last_layer.weight", "last_layer.bias"]
        model = PyTorchForecast(
            "MultiAttnHeadSimple",
            self.keag_file,
            self.keag_file,
            self.keag_file,
            self.model_params)
        result = model.model(torch.rand(1, 20, 3))
        self.assertEqual(result.shape[1], 7)

    def test_train_loss(self):
        print("Now begining train loss test")
        total_loss = torch_single_train(
            self.dummy_model,
            self.opt,
            self.criterion,
            self.data_loader,
            None,
            None,
            False)
        self.assertGreater(total_loss, 100)
        self.assertGreater(total_loss, 752, 000)
        self.assertLess(total_loss, 802000)

    # def test_train_full_transformer(self):
    #     print("Now begining transformer tests")
    #     train_transformer_style(self.transformer, self.full_transformer_params["training_params"], True)
    #     self.assertEqual(1, 1)
    #
    # def test_transfom_validation(self):
    #     # TODO add
    #     pass

    def linear_model_test(self):
        train_transformer_style(
            self.simple_linear_model,
            self.simple_param["training_params"],
            True)

    def test_ae(self):
        model = PyTorchForecast("DARNN", self.keag_file, self.keag_file, self.keag_file, self.meta_model_params)
        for parameter in model.model.parameters():
            self.assertTrue(parameter.requires_grad)

    def test_compute_loss(self):
        crit = self.model.crit[0]
        loss = compute_loss(torch.ones(2, 20), torch.zeros(2, 20), torch.rand(3, 20, 1), crit, None, None)
        self.assertEqual(loss.item(), 1.0)

    def test_test_data(self):
        _, trg = self.model.test_data[0]
        _, trg1 = self.dummy_model.test_data[1]
        _, trg2 = self.transformer.test_data[0]
        _, trg3 = self.simple_linear_model.test_data[0]
        self.assertEqual(trg.shape[0], 20)
        self.assertEqual(trg1.shape[0], 15)
        self.assertEqual(trg2.shape[0], 15)
        self.assertEqual(trg3.shape[0], 25)

    def test_handle_meta(self):
        with open(os.path.join(os.path.dirname(__file__), "da_meta.json")) as f:
            json_config = json.load(f)
        model = PyTorchForecast("DARNN", self.keag_file, self.keag_file, self.keag_file, json_config)
        meta_models, meta_reps, loss = handle_meta_data(model)
        self.assertIsNone(loss)
        self.assertIsInstance(meta_reps, torch.Tensor)
        self.assertIsInstance(meta_models, PyTorchForecast)

    def test_handle_meta2(self):
        with open(os.path.join(os.path.dirname(__file__), "da_meta.json")) as f:
            json_config = json.load(f)
        json_config["meta_data"]["meta_loss"] = "MSE"
        model = PyTorchForecast("DARNN", self.keag_file, self.keag_file, self.keag_file, json_config)
        meta_models, meta_reps, loss = handle_meta_data(model)
        self.assertIsNotNone(loss)
        self.assertIsInstance(meta_reps, torch.Tensor)
        self.assertIsInstance(meta_models, PyTorchForecast)

    def test_scaling_data(self):
        scaled_src, _ = self.model.test_data[0]
        data_unscaled = self.model.test_data.original_df.iloc[0:20]["cfs"].values
        print("shape bw")
        print(scaled_src[:, 0].shape)
        inverse_scale = self.model.test_data.inverse_scale(scaled_src[:, 0])
        self.assertAlmostEqual(inverse_scale.numpy()[0, 0], data_unscaled[0])
        self.assertAlmostEqual(inverse_scale.numpy()[0, 9], data_unscaled[9])

    def test_compute_loss_no_scaling(self):
        exam = torch.Tensor([4.0]).repeat(2, 20, 5)
        exam2 = torch.Tensor([1.0]).repeat(2, 20, 5)
        exam11 = torch.Tensor([4.0]).repeat(2, 20)
        exam1 = torch.Tensor([1.0]).repeat(2, 20)
        d = DilateLoss()
        compute_loss(exam11, exam1, torch.rand(1, 20), d, None)
        # compute_loss(exam, exam2, torch.rand(2, 20), DilateLoss(), None)
        result = compute_loss(exam, exam2, torch.rand(2, 20), torch.nn.MSELoss(), None)
        self.assertEqual(float(result), 9.0)

    def test_z_inf(self):
        train_transformer_style(self.inf, self.inf_params3["training_params"], False)

if __name__ == '__main__':
    unittest.main()
