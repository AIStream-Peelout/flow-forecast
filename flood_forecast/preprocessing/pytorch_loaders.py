from torch.utils.data import Dataset
import pandas as pd

class CSVDataLoader(Dataset):
    def __init__(self, file_path:str, history_length:int, forecast_length:int, target_col:str, 
                 relevant_cols:List, transformations=None):
        """
        A data loader that takes a CSV file
        and returns both the history and the target properly batched 
        """
        super().__init__()
        self.forecast_history = history_length
        self.forecast_length = forecast_length 
        self.df = pd.read_csv(file_path)[relevant_cols]
        self.targ_col = target_col
        
    def __getitem__(self, idx):
        rows = self.df.iloc[idx:self.forecast_history+idx]
        targs_idx_start = self.forecast_history+idx+1
        targ_rows = self.df.iloc[targs_idx_start:self.forecast_length+targs_idx_start]
        src_data = rows.to_numpy()
        src_data = torch.from_numpy(src_data)
        trg_dat = targ_rows.to_numpy()
        trg_dat = torch.from_numpy(trg_dat)
        return src_data, trg_dat
   
     def __len__(self):
        return len(self.df.index)
        
