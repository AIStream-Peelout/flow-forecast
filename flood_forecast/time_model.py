from abc import ABC
from typing import Dict
class TimeSeriesModel(ABC):
    """
    An abstract class used to handle different configurations 
    of models + hyperparams for training, test, and predict functions. 
    This class assumes that data is already split into test train 
    and validation at this point.
    """
    def __init__(self, model_base, training_data, validation_data, test_data):
        self.model = load_model()
        self.training = make_data_load(training_data)
        self.validation = make_data_load(validation_path)
        self.test_loader = make_data_load(test_data)
        
    @abc.abstractmethod
    def load_model(self, model_base, **kwargs) -> object:
        """
        This function should load and return the model 
        this will vary based on the underlying framework used
        """
        pass 
    
    @abc.abstractmethod
    def make_data_load(self, data_path, **kwargs) -> object:
        """
        Intializes a data loader based on
        the provided data path. This may be as simple 
        as a pandas dataframe or as complex as a custom PyTorch data loader
        """
        
    
class PyTorchForecast(TimeSeriesModel):
    def __init__(self, model_base, training_data, test_data, params_dict, weight_path=None):
        super().__init__(model_base, training_data, test_data)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       if self.weight_path is not None:
            model.load_state_dict(torch.load(weight_path, map_location=self.device)
                                  
    def load_model(self, model_base, model_params:Dict):
        pass
                                  
    def train():
        pass
