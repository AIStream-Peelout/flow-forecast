from abc import ABC, abstractmethod
from typing import Dict
import torch 
import json, os
from datetime import datetime
from flood_forecast.model_dict_function import pytorch_model_dict
from flood_forecast.preprocessing.pytorch_loaders import CSVDataLoader

class TimeSeriesModel(ABC):
    """
    An abstract class used to handle different configurations 
    of models + hyperparams for training, test, and predict functions. 
    This class assumes that data is already split into test train 
    and validation at this point.
    """
    def __init__(self, model_base:str, training_data:str, validation_data:str, test_data:str, params:Dict):
        self.model = self.load_model(model_base, params["model_params"])
        self.params = params
        self.training = self.make_data_load(training_data, params["dataset_params"])
        self.validation = self.make_data_load(validation_data, params["dataset_params"])
        self.test_data = self.make_data_load(test_data, params["dataset_params"])
        
    @abstractmethod
    def load_model(self, model_base:str, model_params) -> object:
        """
        This function should load and return the model 
        this will vary based on the underlying framework used
        """
        raise NotImplementedError 
    
    @abstractmethod
    def make_data_load(self, data_path, params:Dict) -> object:
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

    def upload_gcs():
        pass
    
        
class PyTorchForecast(TimeSeriesModel):
    def __init__(self, model_base, training_data, validation_data, test_data, params_dict, weight_path=None):
        super().__init__(model_base, training_data, validation_data, test_data, params_dict)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            
    def load_model(self, model_base:str, model_params:Dict):
        # Load model here 
        if model_base in pytorch_model_dict:
            model = pytorch_model_dict[model_base](**model_params)
        else: 
            raise "Error the model " + model_base + " was not found in the model dict. Please add it."
        return model
    
    def save_model(self, final_path:str):
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        torch.save(self.model.state_dict(), os.path.join(final_path, datetime.now().strftime("%d_%B_%Y%I_%M%p") + "_model.pth"))
        with open(os.path.join(final_path, datetime.now().strftime("%d_%B_%Y_%I_%M%p")) + ".json", "w+") as p:
            json.dump(self.params, p)
    
    def make_data_load(self, data_path:str, dataset_params:Dict):
        if dataset_params["class"] == "default":
            l = CSVDataLoader(data_path, dataset_params["history"], dataset_params["forecast_length"], 
            dataset_params["target_col"], dataset_params["relevant_cols"])
        else:
            # TODO support custom UDL 
            l = None
        return l
                        
