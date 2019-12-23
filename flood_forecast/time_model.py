from abc import ABC, abstractmethod
from typing import Dict
import torch 
import json, os
from datetime import datetime
from flood_forecast.model_dict_function import pytorch_model_dict
from flood_forecast.preprocessing.pytorch_loaders import CSVDataLoader
from flood_forecast.gcp_integration.basic_utils import get_storage_client, upload_file

class TimeSeriesModel(ABC):
    """
    An abstract class used to handle different configurations 
    of models + hyperparams for training, test, and predict functions. 
    This class assumes that data is already split into test train 
    and validation at this point.
    """
    def __init__(self, model_base: str, training_data: str, validation_data: str, test_data:str, params:Dict):
        self.model = self.load_model(model_base, params["model_params"])
        self.params = params
        self.training = self.make_data_load(training_data, params["dataset_params"], "train")
        self.validation = self.make_data_load(validation_data, params["dataset_params"], "valid")
        self.test_data = self.make_data_load(test_data, params["dataset_params"], "test")
        if "GCS" in self.params:
            self.gcs_client = get_storage_client()
        else:
            self.gcs_client = None
        self.wandb = self.wandb_init()
            
    @abstractmethod
    def load_model(self, model_base:str, model_params) -> object:
        """
        This function should load and return the model 
        this will vary based on the underlying framework used
        """
        raise NotImplementedError 
    
    @abstractmethod
    def make_data_load(self, data_path, params:Dict, loader_type:str) -> object:
        """
        Intializes a data loader based on the provided data path. 
        This may be as simple as a pandas dataframe or as complex as 
        a custom PyTorch data loader.
        """
        raise NotImplementedError
        
    @abstractmethod
    def save_model(self, output_path:str):
        """
        Saves a model to a specific path along with a configuration report 
        of the parameters, data, and 
        """
        raise NotImplementedError

    def upload_gcs(self, save_path:str, name, bucket_name=None):
        """
        Function to upload model checkpoints to GCS
        """
        if bucket_name is None:
            bucket_name = os.environ["MODEL_BUCKET"]
        if self.gcs_client:
            upload_file(bucket_name, save_path, "experiments/ " + name, self.gcs_client)
            if self.wandb:
                wandb.config.gcs_path = save_path + "experiments/ " + name
        
    def wandb_init(self):
        if self.params["wandb"] != False:
            import wandb
            wandb.init(config=self.params, name=self.params["wandb"]["name"], tags=self.params["wandb"]["tags"])
            return True 
        return False
    
        
class PyTorchForecast(TimeSeriesModel):
    def __init__(self, model_base, training_data, validation_data, test_data, params_dict, weight_path=None):
        super().__init__(model_base, training_data, validation_data, test_data, params_dict)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            
    def load_model(self, model_base: str, model_params: Dict):
        # Load model here 
        if model_base in pytorch_model_dict:
            model = pytorch_model_dict[model_base](**model_params)
        else: 
            raise Exception("Error the model " + model_base + " was not found in the model dict. Please add it.")
        return model
    
    def save_model(self, final_path: str):
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
        self.upload_gcs(model_save_path, model_name)
        self.upload_gcs(params_save_path, params_name)
    
    def make_data_load(self, data_path: str, dataset_params: Dict, loader_type):
        start_end_params = {}
        if loader_type + "_start" in dataset_params:
            start_end_params["start_stamp"] = dataset_params[loader_type + "_start"]
        if loader_type + "_end" in dataset_params:
            start_end_params["end_stamp"] = dataset_params[loader_type + "_end"] 
        if dataset_params["class"] == "default":
            l = CSVDataLoader(data_path, dataset_params["forecast_history"], dataset_params["forecast_length"],
            dataset_params["target_col"], dataset_params["relevant_cols"], **start_end_params)
        else:
            # TODO support custom Daa Loader
            l = None
        return l
                        
