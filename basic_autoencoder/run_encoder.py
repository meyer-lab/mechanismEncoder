import argparse

from pytorch_lightning.trainer import Trainer

from data.me_loader import MELoader
from encoder.pytorch_encoder import NMEncoder


def main(parser):
    # Instances training dataset
    data_train = MELoader(parser.train)

    # Instances testing dataset
    data_test = MELoader(parser.test)

    # Instances non-mechanistic autoencoder
    feats = data_train.data.shape[1]
    encoder = NMEncoder(feats, feats // 2)

    # Instances PyTorch Lightning trainer
    trainer = Trainer(gpus=0, num_nodes=1, max_epochs=10)

    # Performs model fitting on training set
    trainer.fit(encoder, data_train)

    # Performs test on testing set
    performance = trainer.test(encoder, data_test)
    print(performance)


def _parse_args():
    parser = argparse.ArgumentParser(description='Instance, train, and test autoencoder.')
    parser.add_argument('-train', dest='train', required=True, help='Path to training data')
    parser.add_argument('-test', dest='test', required=True, help='Path to testing data')
    parser.add_argument('-epochs', dest='epochs', default=10, type=int, help='Training epochs')
    parser = parser.parse_args()
    return parser


if __name__ == "__main__":
    parser = _parse_args()
    main(parser)
