import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class MELoader(Dataset):
    """
    Basic PyTorch DataLoader designed for importing simulated mechanistic data.
    """
    def __init__(self, filepath):
        """
        Args:
            filepath (string): Path to the .csv file with annotations.
        """
        self.data = reformat_csv(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]
        return np.array(sample)


def reformat_csv(filepath):
    """
    Re-formats .csv of simulated mechanism data into a pandas DataFrame

    Parameters:
        filepath (str or path): Path to .csv file

    Returns:
        DataFrame containing .csv data re-formatted where columns are features
        and rows are collection time points.
    """
    data = pd.read_csv(filepath, index_col=0)
    sites = list(set(data['site']))
    samples = list(set(data['Sample']))

    df = pd.DataFrame(index=samples)
    for site in sites:
        df[site] = data.loc[site == data['site']]['LogFoldChange'].reset_index(drop=True)

    return df
