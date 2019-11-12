from abc import ABC

class TimeSeriesModel(ABC):
    def __init__(self, model_base, training_data, validation_data, test_data):
        self.model = load_model()
        self.training_data_path = training_data 
        self.validation_data_path = validation_path
        self.test_data = test_data
        
    @abstractmethod
    def load_model(model_path:str) -> object:
        """
        This function should load and return the model 
        this will vary based on the underlying framework used
        """
        pass 
    
class PyTorchForecast(TimeSeriesModel):
    def __init__(self, model_base, training_data, test_data, params_dict, weight_path=None):
        super().__init__(model_base, training_data, test_data)
       if self.weight_path is not None:
          # TODO implement weight loading
          pass 
        
    def train():
        pass
