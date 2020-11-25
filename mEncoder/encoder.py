"""
Materials for a simple linear encoder, and its analytical reverse.
"""

import theano.tensor as T
import theano
import numpy as np
from typing import List

TheanoFunction = theano.compile.function_module.Function

class dA:
    """A simple linear autoencoder. """

    def __init__(self, input_data, n_hidden=1, n_params=12,
                 par_modulation_scale=4):
        self.n_visible = input_data.shape[1]
        assert n_hidden < self.n_visible
        assert n_hidden <= n_params
        assert input_data.ndim == 2
        self.data = input_data
        self.n_hidden = n_hidden
        self.n_params = n_params
        self.n_encode_weights = self.n_visible * self.n_hidden
        self.n_inflate_weights = self.n_hidden * self.n_params
        self.n_inflate_bias = self.n_params
        self.n_encoder_pars = self.n_encode_weights + \
            self.n_inflate_weights + self.n_inflate_bias

        self.encoder_pars = T.specify_shape(T.vector('encoder_pars'),
                                            (self.n_encoder_pars,))

        self.par_modulation_scale = par_modulation_scale

        self.x_names = [
            f'ecoder_{iw}_weight' for iw in range(self.n_encode_weights)
        ] + [
            f'inflate_{iw}_weight' for iw in range(self.n_inflate_weights)
        ] + [
            f'inflate_{iw}_bias' for iw in range(self.n_inflate_bias)
        ]

    def encode(self, pIn):
        """ Run the input through the encoder. """
        W = T.reshape(pIn[0:self.n_encode_weights],
                      (self.n_visible, self.n_hidden))
        return T.dot(self.data, W)

    def inflate_params(self, embedded_data, pIn):
        """ Inflate the input to parameters. """
        W_p = T.reshape(pIn[self.n_encode_weights:
                            self.n_encode_weights+self.n_inflate_weights],
                        (self.n_hidden, self.n_params))
        bias = pIn[-self.n_inflate_bias:]
        return T.nnet.sigmoid(T.dot(embedded_data, W_p) + bias) \
            * self.par_modulation_scale*2 - self.par_modulation_scale

    def encode_params(self, pIn):
        """ Run the encoder and then inflate to parameters. """
        return self.inflate_params(self.encode(pIn), pIn)
    
    def decode(self, embedded_data, pIn):
        """ Run the input through the analytical decoder. """
        W = T.reshape(pIn[0:self.n_encode_weights],
                      (self.n_visible, self.n_hidden))
        return T.dot(embedded_data, T.nlinalg.pinv(W))

    def compile_inflate_pars(self) -> TheanoFunction:
        """
        Compile a theano function that computes the inflated parameters
        """
        return theano.function(
            [self.encoder_pars],
            self.encode_params(self.encoder_pars)
        )

    def compute_inflated_pars(self,
                              encoder_pars: np.ndarray) -> List:
        """
        Compute the inflated parameters
        """
        return self.compile_inflate_pars()(encoder_pars)
