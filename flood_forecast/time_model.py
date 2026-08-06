from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
import json
import os
from datetime import datetime
from flood_forecast.model_dict_function import pytorch_model_dict
from flood_forecast.pre_dict import scaler_dict
from flood_forecast.preprocessing.pytorch_loaders import (
    CSVDataLoader, AEDataloader, TemporalLoader, CSVSeriesIDLoader,
    GeneralClassificationLoader, VariableSequenceLength)
from flood_forecast.gcp_integration.basic_utils import get_storage_client, upload_file
from flood_forecast.utils import make_criterion_functions
from flood_forecast.preprocessing.buil_dataset import get_data
import wandb


def resolve_torch_device(requested_device: str = "auto") -> torch.device:
    """Resolve an explicit or automatic PyTorch compute device.

    Automatic selection preserves the historical CUDA-then-CPU behavior. Apple's Metal
    Performance Shaders backend remains available as an explicit choice so existing callers
    that pass CPU tensors directly to ``model.model`` are not unexpectedly broken. Explicit
    accelerator requests fail instead of silently falling back to CPU.

    :param requested_device: ``auto``, ``cpu``, ``mps``, or a CUDA device such as ``cuda:0``.
    :type requested_device: str
    :return: Resolved PyTorch device.
    :rtype: torch.device
    :raises RuntimeError: If an explicitly requested accelerator is unavailable.
    :raises ValueError: If the device string is unsupported.
    """
    requested_device = requested_device.lower()
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS was requested but is unavailable in this process.")
        return torch.device("mps")
    if requested_device == "cuda" or requested_device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA was requested but is unavailable in this process.")
        return torch.device(requested_device)
    raise ValueError("Unsupported PyTorch device %r." % requested_device)


class TimeSeriesModel(ABC):
    """
    An abstract class used to handle different configurations of models + hyperparams for training, test, and predict
    functions. This class assumes that data is already split into test train and validation at this point.
    """

    def __init__(
            self,
            model_base: str,
            training_data: str,
            validation_data: str,
            test_data: str,
            params: Dict):
        """
        Initializes the TimeSeriesModel class with certain attributes.

        :param model_base: The name of the model to load. This MUST be a key in the model_dic
        model_dict_function.py.
        :type model_base: str
        :param training_data: The path to the training data file
        :type training_data: str
        :param validation_data: The path to the validation data file
        :type validation_data: str
        :param test_data: The path to the test data file
        :type test_data: str
        :param params: A dictionary of parameters to pass to the model, including model_params and dataset_params.
        :type params: Dict
        :return: None
        :rtype: None
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
        self.crit = self.make_metrics(params["metrics"])

    def make_metrics(self, metrics) -> list:
        """Construct backend-specific evaluation metrics.

        The base implementation preserves the historical PyTorch behavior. Other framework
        wrappers override this hook so :class:`TimeSeriesModel` does not instantiate losses from
        the wrong backend.

        :param metrics: Configured metric names or metric parameter mapping.
        :type metrics: list or dict
        :return: Initialized metric callables.
        :rtype: list
        """
        return make_criterion_functions(metrics)

    @abstractmethod
    def load_model(self, model_base: str, model_params: Dict, weight_path=None) -> object:
        """
        This function should load and return the model. This will vary based on the underlying framework used.

        :param model_base: The name of the model to load. This should be a key in the model_dict.
        :type model_base: str
        :param model_params: A dictionary of parameters to pass to the model's constructor.
        :type model_params: Dict
        :param weight_path: The path to the weights to load for a pre-trained model, defaults to None.
        :type weight_path: str, optional
        :return: An instance of the loaded model.
        :rtype: object
        """
        raise NotImplementedError

    @abstractmethod
    def make_data_load(self, data_path: str, params: Dict, loader_type: str) -> object:
        """
        Initializes a data loader based on the provided data_path and parameters.

        This may be as simple as a pandas dataframe or as complex as a custom PyTorch data loader.

        :param data_path: The path to the data file.
        :type data_path: str
        :param params: A dictionary of parameters for the dataset and data loader.
        :type params: Dict
        :param loader_type: A string indicating the type of data being loaded (e.g., "train", "valid", "test").
        :type loader_type: str
        :return: An initialized data loading object.
        :rtype: object
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, output_path: str):
        """Saves a model to a specific path along with a configuration report of the parameters and data info.

        :param output_path: The path to save the model to (should be a directory).
        :type output_path: str
        :return: None
        :rtype: None
        """
        raise NotImplementedError

    def upload_gcs(self, save_path: str, name: str, file_type: str, epoch: int = 0, bucket_name: str = None) -> None:
        """
        Function to upload model checkpoints to GCS.

        :param save_path: The local path of the file to save to GCS.
        :type save_path: str
        :param name: The name you want to save the file as in GCS.
        :type name: str
        :param file_type: The type of file you are saving (e.g., "_model", "_params").
        :type file_type: str
        :param epoch: The epoch number that saving occurred at, defaults to 0.
        :type epoch: int, optional
        :param bucket_name: The name of the bucket to save the file to on GCS, defaults to None.
        :type bucket_name: str, optional
        :return: None
        :rtype: None
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
        """
        Initializes wandb if the params dict contains the "wandb" key or if "sweep" is present.

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
            training_data: str,
            validation_data: str,
            test_data: str,
            params_dict: Dict):
        """
        Initializes the PyTorchForecast class, setting up the device and calling the parent constructor.

        :param model_base: The name of the model to load.
        :type model_base: str
        :param training_data: The path to the training data file.
        :type training_data: str
        :param validation_data: The path to the validation data file.
        :type validation_data: str
        :param test_data: The path to the test data file.
        :type test_data: str
        :param params_dict: A dictionary of parameters to pass to the model and dataset.
        :type params_dict: Dict
        :return: None
        :rtype: None
        """
        self.device = resolve_torch_device(params_dict.get("device", "auto"))
        super().__init__(model_base, training_data, validation_data, test_data, params_dict)
        print("Torch is using " + str(self.device))
        if "weight_path_add" in params_dict:
            self.__freeze_layers__(params_dict["weight_path_add"])

    def __freeze_layers__(self, params: Dict) -> None:
        """
        Function to freeze layers in the model based on parameters.

        :param params: A dictionary containing the "frozen_layers" key with a list of layer names to freeze.
        :type params: Dict
        :return: None
        :rtype: None
        """
        if "frozen_layers" in params:
            print("Layers being frozen")
            for layer in params["frozen_layers"]:
                self.model._modules[layer].requires_grad = False
                for parameter in self.model._modules[layer].parameters():
                    parameter.requires_grad = False

    def load_model(self, model_base: str, model_params: Dict, weight_path: str = None,
                   strict: bool = True) -> torch.nn.Module:
        """
        Loads a PyTorch model, optionally loads weights, and moves it to the appropriate device.

        :param model_base: The name of the model to load, must be a key in pytorch_model_dict.
        :type model_base: str
        :param model_params: A dictionary of parameters to pass to the model's constructor.
        :type model_params: Dict
        :param weight_path: The path to the weights to load, defaults to None.
        :type weight_path: str, optional
        :param strict: Whether state dictionary keys must exactly match model keys, defaults to True.
        :type strict: bool, optional
        :return: The loaded PyTorch model.
        :rtype: torch.nn.Module
        :raises Exception: If the model_base is not found in pytorch_model_dict.
        """
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
        Function to save a PyTorch model's state dictionary and its configuration parameters to a given file path.

        It also handles uploading to GCS and logging the save path to W&B if configured.

        :param final_path: The directory path to save the model and parameters.
        :type final_path: str
        :param epoch: The current epoch number.
        :type epoch: int
        :return: None
        :rtype: None
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
            try:
                wandb.config.save_path = model_save_path
            except Exception as e:
                print("Wandb stupid error")
                print(e.__traceback__)

    def __re_add_params__(self, start_end_params: Dict, dataset_params: Dict, data_path: str) -> Dict:
        """
        Function to re-add the data path and core dataset parameters to the start_end_params dictionary.

        This is used for certain data loaders that need these parameters.

        :param start_end_params: The dictionary containing start/end timestamps and other optional parameters.
        :type start_end_params: Dict
        :param dataset_params: The full dictionary of dataset configuration parameters.
        :type dataset_params: Dict
        :param data_path: The file path to the data.
        :type data_path: str
        :return: The updated start_end_params dictionary.
        :rtype: Dict
        """
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
            the_class: str = "default") -> object:
        """
        Initializes a PyTorch data loader based on the provided data_path and dataset parameters.

        The specific loader class is determined by the "class" key in dataset_params.

        :param data_path: The path to the data file.
        :type data_path: str
        :param dataset_params: A dictionary of parameters for the dataset and data loader.
        :type dataset_params: Dict
        :param loader_type: A string indicating the type of data being loaded ("train", "valid", or "test").
        :type loader_type: str
        :param the_class: The name of the data loader class to use (e.g., "default", "AutoEncoder", "TemporalLoader").
            This is overridden by dataset_params["class"], defaults to "default".
        :type the_class: str, optional
        :return: An initialized PyTorch data loader object.
        :rtype: object
        """
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


def _import_mlx():
    """Import the optional MLX runtime with an actionable installation error.

    :return: ``(mlx.core, mlx.nn)``.
    :rtype: tuple
    :raises ImportError: If the optional MLX dependency is unavailable.
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise ImportError(
            "MLX support requires the optional dependency. Install it with "
            "`pip install flow-forecast[mlx]` or `pip install mlx`."
        ) from exc
    return mx, nn


def resolve_mlx_device(mx, requested_device: str = "auto") -> Any:
    """Set and return the requested MLX default device.

    :param mx: Imported ``mlx.core`` module.
    :type mx: module
    :param requested_device: ``auto``, ``gpu``, or ``cpu``.
    :type requested_device: str
    :return: Resolved MLX device.
    :rtype: mlx.core.Device
    :raises RuntimeError: If GPU was requested but no MLX GPU is available.
    :raises ValueError: If the requested device name is unsupported.
    """
    requested_device = requested_device.lower()
    try:
        gpu_count = mx.device_count(mx.gpu)
    except RuntimeError:
        gpu_count = 0
    if requested_device == "auto":
        requested_device = "gpu" if gpu_count > 0 else "cpu"
    if requested_device == "gpu":
        if gpu_count < 1:
            raise RuntimeError("MLX GPU was requested but is unavailable in this process.")
        mx.set_default_device(mx.gpu)
    elif requested_device == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        raise ValueError("Unsupported MLX device %r; choose auto, gpu, or cpu." % requested_device)
    return mx.default_device()


def _to_mlx_tree(mx, value):
    """Recursively convert tensors and arrays in a Python structure to MLX arrays."""
    if isinstance(value, torch.Tensor):
        return mx.array(value.detach().cpu().numpy())
    if isinstance(value, tuple):
        return tuple(_to_mlx_tree(mx, item) for item in value)
    if isinstance(value, list):
        return [_to_mlx_tree(mx, item) for item in value]
    if isinstance(value, dict):
        return {key: _to_mlx_tree(mx, item) for key, item in value.items()}
    if hasattr(value, "__array__"):
        return mx.array(value)
    return value


class MLXDatasetAdapter:
    """Expose an existing Flow Forecast dataset with native MLX array samples.

    Preprocessing and scaling remain shared with the mature PyTorch data loaders. Conversion is
    lazy at ``__getitem__`` time, so constructing a large dataset does not duplicate its arrays.

    :param dataset: Existing Flow Forecast dataset instance.
    :type dataset: object
    :param mx: Imported ``mlx.core`` module.
    :type mx: module
    """

    def __init__(self, dataset: object, mx):
        self.dataset = dataset
        self.mx = mx

    def __len__(self) -> int:
        """Return the number of available windows."""
        return len(self.dataset)

    def __iter__(self):
        """Iterate over exactly the samples reported by the wrapped dataset."""
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index, *args, **kwargs):
        """Return a recursively converted MLX sample."""
        return _to_mlx_tree(self.mx, self.dataset.__getitem__(index, *args, **kwargs))

    def __getattr__(self, name: str):
        """Delegate scaler and metadata access to the wrapped dataset."""
        return getattr(self.dataset, name)


class MLXForecast(TimeSeriesModel):
    """Framework wrapper for native MLX forecasting models.

    MLX models are distinct implementations registered in
    :mod:`flood_forecast.mlx_model_dict_function`; PyTorch modules are not implicitly converted.
    The wrapper shares repository preprocessing, lazily returns MLX arrays, selects an MLX device,
    and supports native MLX weight loading and saving.
    """

    def __init__(
            self,
            model_base: str,
            training_data: str,
            validation_data: str,
            test_data: str,
            params_dict: Dict):
        """Initialize a native MLX model and its datasets.

        :param model_base: Registered MLX model name.
        :type model_base: str
        :param training_data: Training dataset path.
        :type training_data: str
        :param validation_data: Validation dataset path.
        :type validation_data: str
        :param test_data: Test dataset path.
        :type test_data: str
        :param params_dict: Model, data, metric, and optional ``device`` configuration.
        :type params_dict: dict
        """
        self.mx, self.mlx_nn = _import_mlx()
        self.device = resolve_mlx_device(self.mx, params_dict.get("device", "auto"))
        super().__init__(model_base, training_data, validation_data, test_data, params_dict)
        print("MLX is using " + str(self.device))
        if "weight_path_add" in params_dict:
            self.__freeze_layers__(params_dict["weight_path_add"])

    def make_metrics(self, metrics) -> list:
        """Construct native MLX metric functions from configuration.

        :param metrics: Metric names or a metric parameter mapping.
        :type metrics: list or dict
        :return: Native MLX loss functions.
        :rtype: list
        """
        from flood_forecast.mlx_model_dict_function import make_mlx_criterion_functions
        return make_mlx_criterion_functions(metrics)

    def load_model(self, model_base: str, model_params: Dict, weight_path: str = None,
                   strict: bool = True):
        """Construct a registered native MLX model and optionally load its weights.

        :param model_base: Name in ``mlx_model_dict``.
        :type model_base: str
        :param model_params: Keyword arguments for the model constructor.
        :type model_params: dict
        :param weight_path: Optional ``.npz`` or ``.safetensors`` checkpoint.
        :type weight_path: str, optional
        :param strict: Require checkpoint keys to match model parameters.
        :type strict: bool
        :return: Initialized MLX module.
        :rtype: mlx.nn.Module
        :raises ValueError: If the registered object is not an MLX module.
        :raises KeyError: If the model name is not registered.
        """
        from flood_forecast.mlx_model_dict_function import mlx_model_dict
        if model_base not in mlx_model_dict:
            raise KeyError(
                "MLX model %s was not found. Register it with register_mlx_model()." % model_base)
        model = mlx_model_dict[model_base](**model_params)
        if not isinstance(model, self.mlx_nn.Module):
            raise ValueError("Registered MLX model %s is not an mlx.nn.Module." % model_base)
        if weight_path:
            if "weight_path_add" in self.params and self.params["weight_path_add"].get("excluded_layers"):
                excluded_layers = tuple(self.params["weight_path_add"]["excluded_layers"])
                weights = self.mx.load(str(weight_path))
                weights = [(name, value) for name, value in weights.items()
                           if not name.startswith(excluded_layers)]
                model.load_weights(weights, strict=False)
            else:
                model.load_weights(str(weight_path), strict=strict)
            print("MLX weights successfully loaded")
        self.mx.eval(model.parameters())
        return model

    def __freeze_layers__(self, params: Dict) -> None:
        """Freeze configured MLX parameter subtrees.

        :param params: Mapping that may contain ``frozen_layers`` names.
        :type params: dict
        :return: None.
        :rtype: None
        """
        if "frozen_layers" in params:
            print("MLX layers being frozen")
            self.model.freeze(keys=params["frozen_layers"], strict=True)

    def __re_add_params__(self, start_end_params: Dict, dataset_params: Dict,
                          data_path: str) -> Dict:
        """Reuse common CSV loader parameter handling from the PyTorch wrapper."""
        return PyTorchForecast.__re_add_params__(self, start_end_params, dataset_params, data_path)

    def make_data_load(self, data_path: str, dataset_params: Dict, loader_type: str,
                       the_class: str = "default") -> object:
        """Build a shared Flow Forecast loader and lazily adapt its samples to MLX.

        :param data_path: Dataset path.
        :type data_path: str
        :param dataset_params: Loader configuration.
        :type dataset_params: dict
        :param loader_type: ``train``, ``valid``, or ``test``.
        :type loader_type: str
        :param the_class: Loader class override retained for API compatibility.
        :type the_class: str
        :return: MLX-adapted dataset.
        :rtype: MLXDatasetAdapter
        """
        dataset = PyTorchForecast.make_data_load(
            self, data_path, dataset_params, loader_type, the_class)
        return MLXDatasetAdapter(dataset, self.mx) if dataset is not None else None

    def to_array(self, value):
        """Convert tensors, NumPy arrays, or nested structures to MLX arrays."""
        return _to_mlx_tree(self.mx, value)

    def predict(self, values):
        """Run an eager, evaluation-mode MLX prediction.

        :param values: Array-like or nested model input.
        :type values: object
        :return: Materialized model output.
        :rtype: object
        """
        self.model.eval()
        values = self.to_array(values)
        output = self.model(values)
        if isinstance(output, tuple):
            self.mx.eval(*output)
        else:
            self.mx.eval(output)
        return output

    def save_model(self, final_path: str, epoch: int) -> None:
        """Save MLX weights and the shared JSON configuration.

        :param final_path: Output directory.
        :type final_path: str
        :param epoch: Epoch associated with the checkpoint.
        :type epoch: int
        :return: None.
        :rtype: None
        """
        os.makedirs(final_path, exist_ok=True)
        time_stamp = datetime.now().strftime("%d_%B_%Y%I_%M%p")
        model_name = time_stamp + "_model.safetensors"
        params_name = time_stamp + ".json"
        model_save_path = os.path.join(final_path, model_name)
        params_save_path = os.path.join(final_path, params_name)
        self.mx.eval(self.model.parameters())
        self.model.save_weights(model_save_path)
        with open(params_save_path, "w+") as params_file:
            json.dump(self.params, params_file)
        self.upload_gcs(params_save_path, params_name, "_params", epoch)
        self.upload_gcs(model_save_path, model_name, "_model", epoch)
        if self.wandb:
            wandb.config.save_path = model_save_path


def scaling_function(start_end_params: Dict, dataset_params: Dict) -> Dict:
    """
    Function to initialize a scaler based on the parameters in the dataset_params dict and add it to start_end_params.

    :param start_end_params: The dictionary containing data loading start/end parameters.
    :type start_end_params: Dict
    :param dataset_params: The dictionary of dataset configuration parameters.
    :type dataset_params: Dict
    :return: The start_end_params dictionary updated with an initialized 'scaling' object if a scaler is specified.
    :rtype: Dict
    """
    if "scaler" in dataset_params:
        in_dataset_params = "scaler"
    elif "scaling" in dataset_params:
        in_dataset_params = "scaling"
    else:
        return start_end_params  # Return original if no scaler specified.
    if "scaler_params" in dataset_params:
        scaler = scaler_dict[dataset_params[in_dataset_params]](**dataset_params["scaler_params"])
    else:
        scaler = scaler_dict[dataset_params[in_dataset_params]]()
    start_end_params["scaling"] = scaler
    return start_end_params
