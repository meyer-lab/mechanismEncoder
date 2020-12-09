"""
Materials for a simple linear encoder, and its analytical reverse.
"""

import theano.tensor as tt
import theano
import numpy as np
from typing import List

TheanoFunction = theano.compile.function_module.Function


class AutoEncoder:
    """
    A simple linear autoencoder.

    :param input_data:
        input data for the encoder

    :param n_hidden:
        number of latent variables

    :param n_params:
        number of parameters that to which the embedding will be inflated to
    """

    def __init__(self,
                 input_data: np.ndarray,
                 n_hidden: int = 1,
                 n_params: int = 12):
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
        self.n_inflate_bias = 0  # self.n_params
        self.n_encoder_pars = self.n_encode_weights + \
            self.n_inflate_weights + self.n_inflate_bias

        self.encoder_pars = tt.specify_shape(tt.vector('encoder_pars'),
                                             (self.n_encoder_pars,))

        # self.par_modulation_scale = par_modulation_scale

        self.x_names = [
            f'ecoder_{iw}_weight' for iw in range(self.n_encode_weights)
        ] + [
            f'inflate_{iw}_weight' for iw in range(self.n_inflate_weights)
        ] + [
            f'inflate_{iw}_bias' for iw in range(self.n_inflate_bias)
        ]

    def encode(self, parameters: tt.vector):
        """
        Run the input through the encoder.

        :param parameters:
            parametrization of full autoencoder
        """
        W = tt.reshape(parameters[0:self.n_encode_weights],
                       (self.n_visible, self.n_hidden))
        return tt.dot(self.data, W)

    def getW(self, pIn):
        return tt.reshape(pIn[0:self.n_encode_weights], 
                          (self.n_visible, self.n_hidden))

    def initialW(self):
        """ Calculate an initial encoder parameter set by PCA. """
        LD = np.linalg.svd(self.data)[2].T
        assert LD.shape[0] == self.n_visible
        return (LD[:, 0:self.n_hidden]).flatten()

    def regularize(self, pIn, l2=0.0, ortho=0.0):
        """ Calculate regularization of encoder. """
        W = self.getW(pIn)
        return l2 * tt.nlinalg.norm(pIn, None) \
            + ortho * tt.nlinalg.norm(tt.dot(W.T, W) \
            - tt.eye(self.n_hidden), None)

    def inflate_params(self, embedded_data, pIn):
        """ Inflate the input to parameters. """
        W_p = tt.reshape(
            parameters[self.n_encode_weights:
                       self.n_encode_weights+self.n_inflate_weights],
            (self.n_hidden, self.n_params)
        )
        # bias = pIn[-self.n_inflate_bias:]
        return tt.dot(embedded_data, W_p)
        #return T.nnet.sigmoid(T.dot(embedded_data, W_p) + bias) \
        #    * self.par_modulation_scale*2 - self.par_modulation_scale

    def encode_params(self, parameters: tt.vector):
        """
        Run the encoder and then inflate to parameters.

        :param parameters:
            parametrization of full autoencoder
        """
        return self.inflate_params(self.encode(parameters), parameters)
    
    def decode(self, embedded_data, parameters: tt.vector):
        """
        Run the input through the analytical decoder.

        :param parameters:
            parametrization of full autoencoder
        """
        W = tt.reshape(parameters[0:self.n_encode_weights],
                       (self.n_visible, self.n_hidden))
        return tt.dot(embedded_data, tt.nlinalg.pinv(W))

    def compile_embedded_pars(self) -> TheanoFunction:
        """
        Compile a theano function that computes the inflated parameters
        """
        return theano.function(
            [self.encoder_pars],
            self.encode(self.encoder_pars)
        )

    def compute_embedded_pars(self,
                              encoder_pars: np.ndarray) -> List:
        """
        Compute the inflated parameters
        """
        return self.compile_embedded_pars()(encoder_pars)

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
