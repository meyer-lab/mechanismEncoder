import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class MELoader(Dataset):
    """
    Basic PyTorch DataLoader designed for importing simulated mechanistic data.
    """
    def __init__(self, data, normalize=False):
        """
        Args:
            data (pandas.DataFrame): DataFrame containing mechanism data
            normalize (boolean, default:False): Whether to normalize features
        """
        if normalize:
            normalized = scale(data)
            data.loc[:, :] = normalized

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]
        return np.array(sample)


def reformat_csv(filepath, sample_id, descriptor_id, value_id, drop_axis=None,
                 drop_thresh=0.5):
    """
    Re-formats .csv of simulated mechanism data into a pandas DataFrame

    Parameters:
        filepath (str or path): Path to .csv file
        sample_id (str): Name of column containing sample IDs
        descriptor_id (str, list): Name(s) of columns defining value ID
        value_id (str): Name of column containing sample values
        drop_axis (int, default:None): Which axis to drop when nan values are
            encountered; 0 drops the sample with nan while 1 drops the feature
            with NaN
        drop_thresh (float): If drop_axis is not None, determines the proportion
            of entries that can be NaN before series is dropped; values in
            [0, 1) are valid

    Returns:
        DataFrame containing .csv data re-formatted where columns are features
        and rows are collection time points.
    """
    if isinstance(descriptor_id, str):
        descriptor_id = [descriptor_id]

    csv_df = pd.read_csv(filepath, dtype=str, index_col=0)
    csv_df = csv_df.dropna(subset=descriptor_id)

    uid_col = csv_df[descriptor_id[0]].copy()
    for col in descriptor_id[1:]:
        uid_col += csv_df[col]

    csv_df.insert(0, 'UID', uid_col)
    csv_df = csv_df.drop_duplicates(subset=['UID'] + [sample_id])
    df = csv_df.pivot(index=sample_id, columns='UID', values=value_id)

    df.columns = np.arange(df.shape[1])

    if drop_axis:
        dim = df.shape[1 - drop_axis]
        limit = dim - int(np.ceil(dim * drop_thresh))
        df = df.dropna(axis=drop_axis, thresh=limit)

    df = df.astype(float)

    return df
