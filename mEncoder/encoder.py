"""
Materials for a simple linear encoder, and its analytical reverse.
"""

import theano.tensor as T
import numpy as np


class dA:
    """A simple linear autoencoder. """

    def __init__(self, input_data, n_hidden=1, n_params=12):
        self.n_visible = input_data.shape[1]
        assert n_hidden < self.n_visible
        assert n_hidden <= n_params
        assert input_data.ndim == 2
        assert n_hidden < self.n_visible
        self.data = input_data
        self.n_hidden = n_hidden
        self.n_params = n_params
        self.n_encode_weights = self.n_visible * self.n_hidden
        self.n_inflate_weights = self.n_hidden * self.n_params
        self.n_inflate_bias = self.n_params
        self.n_encoder_pars = self.n_encode_weights + \
            self.n_inflate_weights + self.n_inflate_bias
        self.x_names = [
            f'ecoder_{iw}_weight' for iw in range(self.n_encode_weights)
        ] + [
            f'inflate_{iw}_weight' for iw in range(self.n_inflate_weights)
        ] + [
            f'inflate_{iw}_bias' for iw in range(self.n_inflate_bias)
        ]

    def getW(self, pIn):
        return T.reshape(pIn[0:self.n_encode_weights], (self.n_visible, self.n_hidden))

    def encode(self, pIn):
        """ Run the input through the encoder. """
        return T.dot(self.data, self.getW(pIn))

    def initialW(self):
        """ Calculate an initial encoder parameter set by PCA. """
        LD = np.linalg.svd(self.data)[2].T
        assert LD.shape[0] == self.n_visible
        return (LD[:, 0:self.n_hidden]).flatten()

    def regularize(self, pIn, l2=0.0, ortho=0.0):
        """ Calculate regularization of encoder. """
        W = self.getW(pIn)
        return l2 * T.nlinalg.norm(pIn, None) + ortho * T.nlinalg.norm(T.dot(W.T, W) - T.eye(self.n_hidden), None)

    def inflate_params(self, embedded_data, pIn):
        """ Inflate the input to parameters. """
        W_p = T.reshape(pIn[self.n_encode_weights:
                            self.n_encode_weights+self.n_inflate_weights],
                        (self.n_hidden, self.n_params))
        bias = pIn[-self.n_inflate_bias:]
        return np.power(
            10, T.nnet.sigmoid(T.dot(embedded_data, W_p) + bias)*2 - 1
        )

    def encode_params(self, pIn):
        """ Run the encoder and then inflate to parameters. """
        return self.inflate_params(self.encode(pIn), pIn)
    
    def decode(self, embedded_data, pIn):
        """ Run the input through the analytical decoder. """
        return T.dot(embedded_data, T.nlinalg.pinv(self.getW(pIn)))
