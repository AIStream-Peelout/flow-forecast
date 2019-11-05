from torch.utils.data import Dataset
import pandas as pd

class CSVDataLoader(Dataset):
    def __init__(self, file_path, forecast_length:int, target_col:str, relevant_cols:List, transformations=None):
        """Assume the data frame has already been limited to
        core columns ()"""
        super().__init__()
        self.forecast_length = forecast_length 
        self.df = pd.read_csv(file_path)[relevant_cols]
        self.targ_col = target_col
        
    def __getitem__(self, idx):
        rows = self.df.iloc[idx:self.forecast_length+idx]
        numpy_data = rows.to_numpy()
        torch_data = torch.from_numpy(numpy_data)
        return torch_data
        
