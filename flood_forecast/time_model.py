from abc import ABC, abstractmethod
from typing import Dict
class TimeSeriesModel(ABC):
    """
    An abstract class used to handle different configurations 
    of models + hyperparams for training, test, and predict functions. 
    This class assumes that data is already split into test train 
    and validation at this point.
    """
    def __init__(self, model_base:str, training_data:str, validation_data:str, test_data:str, params:Dict):
        self.model = load_model(model_base, params["model_params"])
        self.params = params
        self.training = make_data_load(training_data, params)
        self.validation = make_data_load(validation_path, params)
        self.test_data = make_data_load(test_data, params)
        
    @abstractmethod
    def load_model(self, model_base, model_params) -> object:
        """
        This function should load and return the model 
        this will vary based on the underlying framework used
        """
        raise NotImplementedError 
    
    @abstractmethod
    def make_data_load(self, data_path, params) -> object:
        """
        Intializes a data loader based on the provided data path. 
        This may be as simple as a pandas dataframe or as complex as 
        a custom PyTorch data loader
        """
        raise NotImplementedError
        
    @abstractmethod
    def save_model(output_path:str):
        """
        Saves a model to a specific path along with a configuration report 
        of the parameters, data, and 
        """
        raise NotImplementedError
    
        
    
class PyTorchForecast(TimeSeriesModel):
    def __init__(self, model_base, training_data, validation_data, test_data, params_dict, weight_path=None):
        super().__init__(model_base, training_data, validation_data, test_data, params_dict)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if weight_path is not None:
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
            
    def load_model(self, model_base, model_params:Dict):
        model_base_to_class = {"PyTorchBasic":PyTorchBasic, "PyTorchTransformer":SimplePositionalEncoding} # Define all PyTorch models here
        model = model_base_to_class[model_base]
        return model
    
    def save_model():
        pass 
    
    def make_data_load():
        pass 
