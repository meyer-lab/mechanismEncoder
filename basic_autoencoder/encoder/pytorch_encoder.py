import logging
import warnings

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.optim import Adam

warnings.filterwarnings("ignore", category=UserWarning)

LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s %(module)s.%(funcName)s:%(lineno)d %(message)s"
)
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    handlers=[logging.FileHandler("pytorch_encoder.log")],
)


class NMEncoder(pl.LightningModule):
    """
    Basic PyTorch Autoencoder. All nodes are linear nodes and are activated via Relu.
    """

    def __init__(self, n_features, n_hidden):
        """
        Parameters:
            n_features (int): Number of features in dataset
            n_hidden (int): Width of hidden layers in autoencoder
        """
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.encoder_in = torch.nn.Linear(in_features=n_features, out_features=n_hidden)
        self.encoder_out = torch.nn.Linear(in_features=n_hidden, out_features=n_hidden)
        self.decoder_in = torch.nn.Linear(in_features=n_hidden, out_features=n_features)
        self.decoder_out = torch.nn.Linear(
            in_features=n_features, out_features=n_features
        )
        self.log = logging.getLogger()
        self.metric = torch.nn.MSELoss()

    def configure_optimizers(self):
        """
        Sets up optimizer for training.

        Returns:
            Adam optimizer for training
        """
        return Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        """
        Feeds input x through the autoencoder.

        Parameters:
            x (PyTorch Tensor): Tensor containing input data

        Returns:
            Output tensor following encoding and decoding of input
        """
        encoded = F.relu(self.encoder_in(x))
        encoded = F.relu(self.encoder_out(encoded))
        decoded = F.relu(self.decoder_in(encoded))
        decoded = F.relu(self.decoder_out(decoded))
        return decoded

    def training_step(self, batch, batch_idx):
        """
        Procedure for training autoencoder.

        Parameters:
            batch (PyTorch Tensor): Batch of training data
            batch_idx (int): Index of batch

        Returns:
            Dictionary mapping loss in training batch
        """
        data = torch.from_numpy(batch)
        data = data.float()

        decoded = self(data)
        decoded = torch.reshape(decoded, data.shape)

        loss = self.metric(data, decoded)
        self.log.info(f"train_loss: {loss}")
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        """
        Procedure for testing autoencoder.

        Parameters:
            batch (PyTorch Tensor): Batch of training data
            batch_idx (int): Index of batch

        Returns:
            Dictionary mapping loss in testing batch
        """
        data = torch.from_numpy(batch)
        data = data.float()

        decoded = self(data)
        decoded = torch.reshape(decoded, data.shape)

        loss = self.metric(data, decoded)
        self.log.info(f"test_loss: {loss}")
        return {"test_loss": loss}
