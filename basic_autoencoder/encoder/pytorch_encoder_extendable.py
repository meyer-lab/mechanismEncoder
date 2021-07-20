import logging
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import Dropout
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
    Basic PyTorch Autoencoder. All nodes are linear nodes and are activated via Sigmoid.
    """

    def __init__(self, n_features, n_hidden, dropout_prob=0.2, n_layers=1, reg_coef=0):
        """
        Parameters:
            n_features (int): Number of features in dataset
            n_hidden (int): Width of hidden layers in autoencoder
        """
        assert n_layers > 0, 'n_layers must be greater than 0'

        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = Dropout(p=dropout_prob)
        self.reg_coef = reg_coef

        widths = [n_features, n_hidden]
        diff = (n_features - n_hidden) // n_layers
        for layer in range(1, n_layers):
            widths.insert(-1, self.n_features - (layer * diff))

        for layer in range(n_layers):
            setattr(self, f'encoder_{layer}', torch.nn.Linear(in_features=widths[layer],
                                                                    out_features=widths[layer + 1]))
            setattr(self, f'decoder_{layer}', torch.nn.Linear(in_features=widths[-1 - layer],
                                                                    out_features=widths[-2 - layer]))

        self.metric = torch.nn.MSELoss()

    def configure_optimizers(self):
        """
        Sets up optimizer for training.

        Returns:
            Adam optimizer for training
        """
        return Adam(self.parameters(), lr=1e-3, weight_decay=self.reg_coef)
        
    def encode_decode(self, x, return_latent=False):
        """
        Runs autoencoder.

        Parameters:
            x (pytorch.Tensor): Tensor containing data to autoencode
            return_latent (bool): Whether to return latent 
                attributes (default: False)

        Return:
            Output tensor following encoding and decoding of input
        """
        for i in range(self.n_layers):
            x = getattr(self, f'encoder_{i}')(x)
            if i != self.n_layers - 1:
                x = F.sigmoid(x)
            x = self.dropout(x)

        if return_latent:
            latent = x.detach().cpu().numpy()

        for i in range(self.n_layers):
            x = getattr(self, f'decoder_{i}')(x)
            x = F.sigmoid(x)
            x = self.dropout(x)
        
        if return_latent:
            return x, latent
        else:
            return x

    def forward(self, x, return_latent=False):
        """
        Feeds input x through the autoencoder.

        Parameters:
            x (PyTorch Tensor): Tensor containing input data

        Returns:
            Output tensor following encoding and decoding of input
            with imputed data
        """
        original = x.detach()
        mask = torch.isnan(original)

        x[mask] = 0
        x = self.encode_decode(x)
        x = torch.where(mask, x, original)

        if return_latent:
            x, latent = self.encode_decode(original, True)
        else:
            x = self.encode_decode(original)
        
        if return_latent:
            return x, latent
        else:
            return x

    def training_step(self, batch, batch_idx):
        """
        Procedure for training autoencoder.

        Parameters:
            batch (PyTorch Tensor): Batch of training data
            batch_idx (int): Index of batch

        Returns:
            Dictionary mapping loss in training batch
        """
        batch = batch.float()
        decoded = self(batch)
        decoded = torch.reshape(decoded, batch.shape)
        loss = self.metric(batch, decoded)

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
        batch = batch.float()
        decoded, latent = self(batch, True)
        decoded = torch.reshape(decoded, batch.shape)
        loss = self.metric(batch, decoded)

        return {"test_loss": loss.item(), "latent": latent}
