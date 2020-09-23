"""
Materials for a simple linear encoder, and its analytical reverse.
"""

import theano
import theano.tensor as T
import numpy as np


class dA:
    """A simple linear autoencoder. """

    def __init__(self, n_visible=50, n_hidden=10, n_params=12, W=None, W_p=None):
        assert n_hidden < n_visible
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_params = n_params

        if W is None:
            initial_W = np.random.normal(size=(n_visible, n_hidden))
            initial_W = np.asarray(initial_W, dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W')

        if W_p is None:
            initial_W_p = np.random.normal(size=(n_hidden, n_params))
            initial_W_p = np.asarray(initial_W_p, dtype=theano.config.floatX)
            W_p = theano.shared(value=initial_W_p, name='W_p')

        self.W = W
        self.W_p = W_p

    def encode(self, input):
        """ Run the input through the encoder. """
        return T.dot(input, self.W)

    def encode_params(self, input):
        """ Run the encoder and then inflate to parameters. """
        return T.dot(self.encode(input), self.W_p)
    
    def decode(self, input):
        """ Run the input through the analytical decoder. """
        return T.dot(input, T.nlinalg.pinv(self.W))
