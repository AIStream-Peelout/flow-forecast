from abc import ABC, abstractmethod
from typing import Dict
import torch
import json
import os
from datetime import datetime
from flood_forecast.model_dict_function import pytorch_model_dict
from flood_forecast.pre_dict import scaler_dict
from flood_forecast.preprocessing.pytorch_loaders import CSVDataLoader, AEDataloader
from flood_forecast.gcp_integration.basic_utils import get_storage_client, upload_file
import wandb


class TimeSeriesModel(ABC):
    """
    An abstract class used to handle different configurations
    of models + hyperparams for training, test, and predict functions.
    This class assumes that data is already split into test train
    and validation at this point.
    """

    def __init__(
            self,
            model_base: str,
            training_data: str,
            validation_data: str,
            test_data: str,
            params: Dict):
        self.params = params
        if "weight_path" in params:
            self.model = self.load_model(model_base, params["model_params"], params["weight_path"])
        else:
            self.model = self.load_model(model_base, params["model_params"])
        params["dataset_params"]["forecast_test_len"] = params["inference_params"]["hours_to_forecast"]
        self.training = self.make_data_load(training_data, params["dataset_params"], "train")
        self.validation = self.make_data_load(validation_data, params["dataset_params"], "valid")
        self.test_data = self.make_data_load(test_data, params["dataset_params"], "test")
        if "GCS" in self.params and self.params["GCS"]:
            self.gcs_client = get_storage_client()
        else:
            self.gcs_client = None
        self.wandb = self.wandb_init()

    @abstractmethod
    def load_model(self, model_base: str, model_params: Dict, weight_path=None) -> object:
        """
        This function should load and return the model
        this will vary based on the underlying framework used
        """
        raise NotImplementedError

    @abstractmethod
    def make_data_load(self, data_path, params: Dict, loader_type: str) -> object:
        """
        Intializes a data loader based on the provided data_path.
        This may be as simple as a pandas dataframe or as complex as
        a custom PyTorch data loader.
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, output_path: str):
        """
        Saves a model to a specific path along with a configuration report
        of the parameters and data info.
        """
        raise NotImplementedError

    def upload_gcs(self, save_path: str, name: str, file_type: str, epoch=0, bucket_name=None):
        """
        Function to upload model checkpoints to GCS
        """
        if self.gcs_client:
            if bucket_name is None:
                bucket_name = os.environ["MODEL_BUCKET"]
            print("Data saved to: ")
            print(name)
            upload_file(bucket_name, os.path.join("experiments", name), save_path, self.gcs_client)
            online_path = os.path.join("gs://", bucket_name, "experiments", name)
            if self.wandb:
                wandb.config.update({"gcs_m_path_" + str(epoch) + file_type: online_path})

    def wandb_init(self):
        if self.params["wandb"]:
            wandb.init(
                project=self.params["wandb"]["project"],
                config=self.params,
                name=self.params["wandb"]["name"],
                tags=self.params["wandb"]["tags"])
            return True
        elif "sweep" in self.params:
            print("Using Wandb config:")
            print(wandb.config)
            return True
        return False


class PyTorchForecast(TimeSeriesModel):
    def __init__(
            self,
            model_base: str,
            training_data,
            validation_data,
            test_data,
            params_dict: Dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(model_base, training_data, validation_data, test_data, params_dict)
        print("Torch is using " + str(self.device))

    def load_model(self, model_base: str, model_params: Dict, weight_path: str = None, strict=True):
        if model_base in pytorch_model_dict:
            model = pytorch_model_dict[model_base](**model_params)
            if weight_path:
                checkpoint = torch.load(weight_path, map_location=self.device)
                if "weight_path_add" in self.params:
                    if "excluded_layers" in self.params["weight_path_add"]:
                        excluded_layers = self.params["weight_path_add"]["excluded_layers"]
                        for layer in excluded_layers:
                            del checkpoint[layer]
                        print("sucessfully deleted layers")
                    strict = False
                model.load_state_dict(checkpoint, strict=strict)
                print("Weights sucessfully loaded")
            model.to(self.device)
            # TODO create a general loop to convert all model tensor params to device
            if hasattr(model, "mask"):
                model.mask = model.mask.to(self.device)
            if hasattr(model, "tgt_mask"):
                model.tgt_mask = model.tgt_mask.to(self.device)
        else:
            raise Exception(
                "Error the model " +
                model_base +
                " was not found in the model dict. Please add it.")
        return model

    def save_model(self, final_path: str, epoch: int) -> None:
        """
        Function to save a model to a given file path
        """
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        time_stamp = datetime.now().strftime("%d_%B_%Y%I_%M%p")
        model_name = time_stamp + "_model.pth"
        params_name = time_stamp + ".json"
        model_save_path = os.path.join(final_path, model_name)
        params_save_path = os.path.join(final_path, time_stamp + ".json")
        torch.save(self.model.state_dict(), model_save_path)
        with open(params_save_path, "w+") as p:
            json.dump(self.params, p)
        self.upload_gcs(params_save_path, params_name, "_params", epoch)
        self.upload_gcs(model_save_path, model_name, "_model", epoch)
        if self.wandb:
            wandb.config.save_path = model_save_path

    def make_data_load(
            self,
            data_path: str,
            dataset_params: Dict,
            loader_type: str,
            the_class="default"):
        start_end_params = {}
        the_class = dataset_params["class"]
        # TODO clean up else if blocks
        if loader_type + "_start" in dataset_params:
            start_end_params["start_stamp"] = dataset_params[loader_type + "_start"]
        if loader_type + "_end" in dataset_params:
            start_end_params["end_stamp"] = dataset_params[loader_type + "_end"]
        if "scaler" in dataset_params:
            start_end_params["scaling"] = scaler_dict[dataset_params["scaler"]]
        if "interpolate" in dataset_params:
            start_end_params["interpolate_param"] = dataset_params["interpolate"]
        is_proper_dataloader = loader_type == "test" and the_class == "deault"
        if is_proper_dataloader and "forecast_test_len" in dataset_params:
            loader = CSVDataLoader(
                data_path,
                dataset_params["forecast_history"],
                dataset_params["forecast_test_len"],
                dataset_params["target_col"],
                dataset_params["relevant_cols"],
                **start_end_params)
        elif the_class == "default":
            loader = CSVDataLoader(
                data_path,
                dataset_params["forecast_history"],
                dataset_params["forecast_length"],
                dataset_params["target_col"],
                dataset_params["relevant_cols"],
                **start_end_params)
        elif the_class == "AutoEncoder":
            loader = AEDataloader(
                data_path,
                dataset_params["relevant_cols"],
                **start_end_params
            )
        else:
            # TODO support custom DataLoader
            loader = None
        return loader
