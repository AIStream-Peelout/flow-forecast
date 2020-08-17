import pandas as pd 
import numpy as np


class TabularDataset(Dataset):
    def __init__(self, data, cat_cols=None, output_col=None):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        ----------

        data: pandas data frame
        The data frame object for the input data. It must
        contain all the continuous, categorical and the
        output columns to be used.

        cat_cols: List of strings
        The names of the categorical columns in the data.
        These columns will be passed through the embedding
        layers in the model. These columns must be
        label encoded beforehand.

        output_col: string
        The name of the output variable column in the data
        provided.
        """

        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + [output_col]]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.n, 1))

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]


class AutoLabels(Dataset):
    """A dataset for getting labels"""

    def __init__(self, file_path: str, device='cpu', test_mode=False, start_index=0, end_index=1000):
        """
        Args:
            file_path (string): Path to H5 file
            device (string): Whether to use GPU or CPU
        """
        self.test = test_mode
        if not test_mode:
          self.features = pd.read_hdf(file_path, "train_x")[start_index:end_index]
          self.labels = pd.read_hdf(file_path, "train_y")[start_index:end_index]
        else: 
          self.features = pd.read_hdf(file_path, "test_x")
        
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
      if self.test:
        return torch.from_numpy(self.features.iloc[idx].to_numpy()).to(self.device)
      return torch.from_numpy(self.features.iloc[idx].to_numpy()).to(self.device).float(), torch.from_numpy(np.array(self.labels.iloc[idx])).to(self.device).float()

