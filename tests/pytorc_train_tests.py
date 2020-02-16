
import os
import torch
from flood_forecast.model_dict_function import pytorch_model_dict as pytorch_model_dict1
from flood_forecast.time_model import PyTorchForecast
from flood_forecast.pytorch_training import torch_single_train
import unittest
from flood_forecast.pytorch_training import train_transformer_style

class PyTorchTrainTests(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_init")
        self.model_params = {"model_params":{"number_time_series":3, "seq_len":20}, 
        "dataset_params":{"forecast_history": 20, "class":"default", "forecast_length":20, "relevant_cols":["cfs", "temp", "precip"], "target_col":["cfs"], "interpolate": False},
        "training_params": {"optimizer":"Adam", "lr":.1, "criterion": "MSE", "epochs":1, "batch_size":2,  "optim_params":{}},
                            "wandb":False}
        self.keag_file = os.path.join(self.test_path, "keag_small.csv")
        self.model = PyTorchForecast("MultiAttnHeadSimple", self.keag_file, self.keag_file, self.keag_file, self.model_params)
        self.dummy_model = PyTorchForecast("DummyTorchModel", self.keag_file, self.keag_file, self.keag_file, {"model_params":{"forecast_length":3}})

    def test_pytorch_train_base(self):
        self.assertEqual(self.model.model.dense_shape.in_features, 3)
        self.assertEqual(self.model.model.multi_attn.embed_dim, 128)

    def test_train_and_resume(self):
        train_transformer_style(self.model, self.model_params["training_params"])
        self.assertEqual(len(os.listdir("model_save")), 2)
        self.assertEqual(1,1)
        model2 = PyTorchForecast("MultiAttnHeadSimple", self.keag_file, self.keag_file, self.keag_file, self.model_params)
        data = torch.rand(2, 20,3)
        self.model_params["weight_path"] = os.path.join("model_save", sorted(os.listdir("model_save"))[1])
        model3 = PyTorchForecast("MultiAttnHeadSimple", self.keag_file, self.keag_file, self.keag_file, self.model_params)
        basic_model = model2.model
        basic_model.eval()
        pre_loaded_model = model3.model
        pre_loaded_model.eval()
        self.assertFalse(torch.allclose(pre_loaded_model(data), basic_model(data)))
        self.assertTrue(torch.allclose(basic_model(data), basic_model(data)))

    def test_train_loss(self):
        pass
if __name__ == '__main__':
    unittest.main()
