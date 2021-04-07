import argparse

import numpy as np
import torch
from pytorch_lightning.trainer import Trainer
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from data.loaders.me_loader import MELoader, reformat_csv
from encoder.pytorch_encoder_extendable import NMEncoder


def run_encoder(train, test, epochs, width, depth, dropout_prob=0.2, reg_coef=0):
    """
    Instances and runs extendable autoencoder.

    Parameters:
        train (pandas.DataFrame): DataFrame of training data
        test (pandas.DataFrame): DataFrame of testing data
        epochs (int): Training epochs
        width (int): Number of latent attributes
        depth (int): Number of encoding/decoding layers
        dropout_prob (float, default=0.2): Probability of drop-out
        reg_coef (float, default=0): Regularization coefficient

    Returns:
        Autoencoder loss on test data
    """
    # Instances training dataset
    data_train = MELoader(train)

    # Instances testing dataset
    data_test = MELoader(test)

    # Instances non-mechanistic autoencoder
    feats = data_train.data.shape[1]
    encoder = NMEncoder(
        feats, width, dropout_prob=dropout_prob, n_layers=depth, reg_coef=reg_coef
    )

    # Instances PyTorch Lightning trainer
    trainer = Trainer(
        gpus=1,
        max_epochs=epochs,
        checkpoint_callback=False,
        logger=False,
        weights_summary=None,
        progress_bar_refresh_rate=0,
    )

    # Performs model fitting on training set
    trainer.fit(encoder, DataLoader(dataset=data_train))

    # Performs test on testing set
    performance = trainer.test(encoder, DataLoader(dataset=data_test))

    return performance[0]["test_loss"]


def main(parser):
    sample_id = "sample"  # Name of sample ID field
    descriptor_ids = ["Protein", "Peptide", "site"]  # Name of sample descriptors
    value_id = "logRatio"  # Name of sample value field
    data = reformat_csv(parser.input, sample_id, descriptor_ids, value_id, drop_axis=1)

    width = parser.width  # Width of latent attribute layer
    depth = parser.layers  # Layers in encoder and decoder
    dropout_prob = parser.dropout  # Dropout probability of dropout layers
    reg_coef = parser.reg_coef  # L2 Regularization coefficient

    fold_means = []
    kf = KFold(n_splits=parser.folds)
    for train_index, test_index in kf.split(data):
        test = data.iloc[test_index, :]
        train = data.iloc[train_index, :]
        loss = run_encoder(
            train, test, width, 20, depth, dropout_prob=dropout_prob, reg_coef=reg_coef
        )
        fold_means.append(loss)

    print(np.mean(fold_means))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-validate with extendable autoencoder"
    )
    parser.add_argument(
        "-d",
        "--dropout",
        dest="dropout",
        default=0.2,
        type=float,
        help="Dropout probability",
    )
    parser.add_argument(
        "-i", "--input", dest="input", required=True, help="Path to AML data"
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", default=10, type=int, help="Training epochs"
    )
    parser.add_argument(
        "-f",
        "--folds",
        dest="folds",
        default=30,
        type=int,
        help="Cross-validation folds",
    )
    parser.add_argument(
        "-l",
        "--layers",
        dest="layers",
        default=1,
        type=int,
        help="Encoder/decoder layers",
    )
    parser.add_argument(
        "-r",
        "--reg_coef",
        dest="reg_coef",
        default=0,
        type=float,
        help="L2 regularization coefficient",
    )
    parser.add_argument(
        "-w",
        "--width",
        dest="width",
        default=10,
        type=int,
        help="Number of latent attributes",
    )
    parser = parser.parse_args()
    return parser


if __name__ == "__main__":
    parser = _parse_args()
    main(parser)
