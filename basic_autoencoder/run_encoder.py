import argparse

import numpy as np
from pytorch_lightning.trainer import Trainer
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader

from data.loaders.me_loader import MELoader, reformat_csv
from encoder.pytorch_encoder import NMEncoder


def run_encoder(train, test, epochs):
    """
    Instances and runs autoencoder.

    Parameters:
        train (pandas.DataFrame): DataFrame of training data
        test (pandas.DataFrame): DataFrame of testing data
        epochs (int): Training epochs

    Returns:
        Autoencoder loss on test data
    """
    # Instances training dataset
    data_train = MELoader(train)

    # Instances testing dataset
    data_test = MELoader(test)

    # Instances non-mechanistic autoencoder
    feats = data_train.data.shape[1]
    encoder = NMEncoder(feats, feats // 2)

    # Instances PyTorch Lightning trainer
    trainer = Trainer(gpus=1, num_nodes=1, max_epochs=epochs)

    # Performs model fitting on training set
    trainer.fit(encoder, DataLoader(dataset=data_train))

    # Performs test on testing set
    performance = trainer.test(encoder, DataLoader(dataset=data_test))

    return performance[0]["test_loss"]


def main(parser):
    sample_id = 'sample'  # Name of sample ID field
    descriptor_ids = ['Protein', 'Peptide', 'site']  # Name of sample descriptors
    value_id = 'logRatio'  # Name of sample value field
    data = reformat_csv(parser.input, sample_id, descriptor_ids, value_id, drop_axis=1)

    width = 100  # Width of latent attribute layer
    depth = 1  # Layers in encoder and decoder

    fold_means = []
    kf = KFold(n_splits=parser.folds)
    for train_index, test_index in kf.split(data):
        test = data.iloc[test_index, :]
        train = data.iloc[train_index, :]
        loss = run_encoder(train, test, parser.epochs)
        fold_means.append(loss)

    print(np.mean(fold_means))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-validate with extendable autoencoder"
    )
    parser.add_argument(
        "-i", "--input", dest="input", required=True, help="Path to AML data"
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", default=10, type=int, help="Training epochs"
    )
    parser.add_argument(
        "-f", "--folds", dest="folds", default=30, type=int, help="Cross-validation folds"
    )
    parser = parser.parse_args()
    return parser


if __name__ == "__main__":
    parser = _parse_args()
    main(parser)
