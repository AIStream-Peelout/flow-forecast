from abc import ABC, abstractmethod
from typing import Dict
import torch
import json
import os
from datetime import datetime
from flood_forecast.model_dict_function import pytorch_model_dict
from flood_forecast.pre_dict import scaler_dict
from flood_forecast.preprocessing.pytorch_loaders import (CSVDataLoader, AEDataloader, TemporalLoader,
                                                          CSVSeriesIDLoader, GeneralClassificationLoader,
                                                          VariableSequenceLength)
from flood_forecast.gcp_integration.basic_utils import get_storage_client, upload_file
from flood_forecast.utils import make_criterion_functions
from flood_forecast.preprocessing.buil_dataset import get_data
import wandb


class TimeSeriesModel(ABC):
    """An abstract class used to handle different configurations of models + hyperparams for training, test, and predict
    functions. This class assumes that data is already split into test train and validation at this point.
    """

    def __init__(
            self,
            model_base: str,
            training_data: str,
            validation_data: str,
            test_data: str,
            params: Dict):
        """Initializes the TimeSeriesModel class with certain attributes.

        :param model_base: The name of the model to load. This MUST be a key in the model_dic
        model_dict_function.py.
        :type model_base: str
        :param training_data: The path to the training data file
        :type training_data: str
        :param validation_data: The path to the validation data file
        :type validation_data: str
        :param test_data: The path to the test data file
        :type test_data: str
        :param params: A dictionary of parameters to pass to the model
        """
        self.params = params
        if "weight_path" in params:
            # If weight_path is present it means we are loading an existing model rather than training from scratch.
            params["weight_path"] = get_data(params["weight_path"])
            self.model = self.load_model(model_base, params["model_params"], params["weight_path"])
        else:
            self.model = self.load_model(model_base, params["model_params"])
        self.training = self.make_data_load(training_data, params["dataset_params"], "train")
        self.validation = self.make_data_load(validation_data, params["dataset_params"], "valid")
        self.test_data = self.make_data_load(test_data, params["dataset_params"], "test")
        if "GCS" in self.params and self.params["GCS"]:
            self.gcs_client = get_storage_client()
        else:
            self.gcs_client = None
        self.wandb = self.wandb_init()
        self.crit = make_criterion_functions(params["metrics"])

    @abstractmethod
    def load_model(self, model_base: str, model_params: Dict, weight_path=None) -> object:
        """This function should load and return the model. This will vary based on the underlying framework used.

        :param model_base: The name of the model to load. This should be a key in the model_dict.
        :type model_base: str
        :param model_params: A dictionary of parameters to pass to the model
        :param weight_path: The path to the weights to load
        """
        raise NotImplementedError

    @abstractmethod
    def make_data_load(self, data_path, params: Dict, loader_type: str) -> object:
        """Initializes a data loader based on the provided data_path.

        This may be as simple as a pandas dataframe or as complex as a custom PyTorch data loader.
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, output_path: str):
        """Saves a model to a specific path along with a configuration report of the parameters and data info.

        :param output_path: The path to save the model to (should be a directory)
        :type output_path: str
        """
        raise NotImplementedError

    def upload_gcs(self, save_path: str, name: str, file_type: str, epoch=0, bucket_name=None):
        """Function to upload model checkpoints to GCS.
        :param save_path: The path of the file to save to GCS.
        :type save_path: str
        :param name: The name you want to save the file as.
        :type name: str
        :param file_type: The type of file you are saving.
        :type file_type: str
        :param epoch: The epoch number that saving occured at.
        :type epoch: int
        :param bucket_name: The name of the bucket to save the file to on GCS.
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

    def wandb_init(self) -> bool:
        """Initializes wandb if the params dict contains the wandb key or if sweep is present.
        :return: True if wandb is initialized, False otherwise.
        :rtype: bool
        """
        if self.params["wandb"]:
            wandb.init(
                id=wandb.util.generate_id(),
                project=self.params["wandb"].get("project"),
                config=self.params,
                name=self.params["wandb"].get("name"),
                tags=self.params["wandb"].get("tags")),
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
        if "weight_path_add" in params_dict:
            self.__freeze_layers__(params_dict["weight_path_add"])

    def __freeze_layers__(self, params: Dict):
        """Function to freeze layers in the model."""
        if "frozen_layers" in params:
            print("Layers being frozen")
            for layer in params["frozen_layers"]:
                self.model._modules[layer].requires_grad = False
                for parameter in self.model._modules[layer].parameters():
                    parameter.requires_grad = False

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
        """Function to save a model to a given file path."""
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
            try:
                wandb.config.save_path = model_save_path
            except Exception as e:
                print("Wandb stupid error")
                print(e.__traceback__)

    def __re_add_params__(self, start_end_params: Dict, dataset_params, data_path):
        """Function to re-add the params to the model."""
        start_end_params["file_path"] = data_path
        start_end_params["forecast_history"] = dataset_params["forecast_history"]
        start_end_params["forecast_length"] = dataset_params["forecast_length"]
        start_end_params["target_col"] = dataset_params["target_col"]
        start_end_params["relevant_cols"] = dataset_params["relevant_cols"]
        return start_end_params

    def make_data_load(
            self,
            data_path: str,
            dataset_params: Dict,
            loader_type: str,
            the_class="default"):
        start_end_params = {}
        the_class = dataset_params["class"]
        start_end_params = scaling_function(start_end_params, dataset_params)
        # TODO clean up else if blocks
        if loader_type + "_start" in dataset_params:
            start_end_params["start_stamp"] = dataset_params[loader_type + "_start"]
        if loader_type + "_end" in dataset_params:
            start_end_params["end_stamp"] = dataset_params[loader_type + "_end"]
        if "interpolate" in dataset_params:
            start_end_params["interpolate_param"] = dataset_params["interpolate"]
        if "feature_param" in dataset_params:
            start_end_params["feature_params"] = dataset_params["feature_param"]
            "Feature param put into stuff"
        if "sort_column" in dataset_params:
            start_end_params["sort_column"] = dataset_params["sort_column"]
        if "scaled_cols" in dataset_params:
            start_end_params["scaled_cols"] = dataset_params["scaled_cols"]
        if "no_scale" in dataset_params:
            start_end_params["no_scale"] = dataset_params["no_scale"]
        if "id_series_col" in dataset_params:
            start_end_params["id_series_col"] = dataset_params["id_series_col"]
        if the_class == "AutoEncoder":
            start_end_params["forecast_history"] = dataset_params["forecast_history"]
            start_end_params["target_col"] = dataset_params["relevant_cols"]
        is_proper_dataloader = loader_type == "test" and the_class == "default"
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
        elif the_class == "TemporalLoader":
            start_end_params = self.__re_add_params__(start_end_params, dataset_params, data_path)
            label_len = 0
            if "label_len" in dataset_params:
                label_len = dataset_params["label_len"]
            loader = TemporalLoader(
                dataset_params["temporal_feats"],
                start_end_params,
                label_len=label_len)
        elif the_class == "SeriesIDLoader":
            start_end_params = self.__re_add_params__(start_end_params, dataset_params, data_path)
            loader = CSVSeriesIDLoader(
                dataset_params["series_id_col"],
                start_end_params,
                dataset_params["return_method"]
            )
        elif the_class == "GeneralClassificationLoader":
            dataset_params["forecast_length"] = 1
            start_end_params = self.__re_add_params__(start_end_params, dataset_params, data_path)
            start_end_params["sequence_length"] = dataset_params["sequence_length"]
            loader = GeneralClassificationLoader(start_end_params, dataset_params["n_classes"])
        elif the_class == "VariableSequenceLength":
            start_end_params = self.__re_add_params__(start_end_params, dataset_params, data_path)
            if "pad_len" in dataset_params:
                pad_le = dataset_params["pad_len"]
            else:
                pad_le = None
            loader = VariableSequenceLength(dataset_params["series_marker_column"], start_end_params,
                                            pad_le, dataset_params["task"])

        else:
            loader = None
        return loader


def scaling_function(start_end_params: Dict, dataset_params: Dict) -> Dict:
    """
    Function to scale the data based on the parameters in the dataset_params dict
    :param start_end_params: The start_end_params dictionary
    :param dataset_params: The dataset_params dictionary
    :return: The start_end_params dictionary
    """
    if "scaler" in dataset_params:
        in_dataset_params = "scaler"
    elif "scaling" in dataset_params:
        in_dataset_params = "scaling"
    else:
        return {}
    if "scaler_params" in dataset_params:
        scaler = scaler_dict[dataset_params[in_dataset_params]](**dataset_params["scaler_params"])
    else:
        scaler = scaler_dict[dataset_params[in_dataset_params]]()
    start_end_params["scaling"] = scaler
    return start_end_params
