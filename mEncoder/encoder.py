"""
Materials for a simple linear encoder, and its analytical reverse.
"""

import aesara.tensor as aet
import aesara
import numpy as np
from typing import List

AFunction = aesara.compile.Function


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
        self.n_encoder_pars = self.n_encode_weights + \
            self.n_inflate_weights

        self.encode_weights = aet.specify_shape(aet.vector('encode_weights'),
                                                (self.n_encode_weights,))

        self.inflate_weights = aet.specify_shape(aet.vector('inflate_weights'),
                                                 (self.n_inflate_weights,))

        self.encoder_pars = aet.specify_shape(aet.vector('encoder_pars'),
                                              (self.n_encoder_pars,))

        # self.par_modulation_scale = par_modulation_scale

        self.x_names = [
            f'ecoder_{iw}_weight' for iw in range(self.n_encode_weights)
        ] + [
            f'inflate_{iw}_weight' for iw in range(self.n_inflate_weights)
        ]

    def encode(self, parameters: aet.vector):
        """
        Run the input through the encoder.

        :param parameters:
            parametrization of full autoencoder
        """
        W = aet.reshape(parameters[0:self.n_encode_weights],
                        (self.n_visible, self.n_hidden))
        return aet.dot(self.data, W)

    def getW(self, parameters):
        return aet.reshape(parameters[0:self.n_encode_weights],
                           (self.n_visible, self.n_hidden))

    def initialW(self):
        """ Calculate an initial encoder parameter set by PCA. """
        LD = np.linalg.svd(self.data)[2].T
        assert LD.shape[0] == self.n_visible
        return (LD[:, 0:self.n_hidden]).flatten()

    def regularize(self, parameters, l2=0.0, ortho=0.0):
        """ Calculate regularization of encoder. """
        W = self.getW(parameters)
        return l2 * aet.nlinalg.norm(parameters, None) \
            + ortho * aet.nlinalg.norm(
            aet.dot(W.T, W) - aet.eye(self.n_hidden), None
        )

    def inflate_params_restricted(self, embedding: np.ndarray,
                                  parameters: aet.vector):
        """ Inflate the input to parameters (partial parameter vector) """
        W_p = aet.reshape(
            parameters, (self.n_hidden, self.n_params)
        )
        return aet.dot(embedding, W_p)

    def inflate_params(self, embedding: np.ndarray, parameters: aet.vector):
        """ Inflate the input to parameters (full parameter vector) """

        return self.inflate_params_restricted(
            embedding,
            parameters[self.n_encode_weights:
                       self.n_encode_weights + self.n_inflate_weights],
        )

    def encode_params(self, parameters: aet.vector):
        """
        Run the encoder and then inflate to parameters.

        :param parameters:
            parametrization of full autoencoder
        """
        return self.inflate_params(self.encode(parameters), parameters)
    
    def decode(self, embedded_data: np.ndarray, parameters: aet.vector):
        """
        Run the input through the analytical decoder.

        :param embedded_data:
            latent embedding of data

        :param parameters:
            parametrization of full autoencoder
        """
        W = aet.reshape(parameters[0:self.n_encode_weights],
                        (self.n_visible, self.n_hidden))
        return aet.dot(embedded_data, aet.nlinalg.pinv(W))

    def compile_embedded_pars(self) -> AFunction:
        """
        Compile a theano function that computes the inflated parameters
        """
        return aesara.function(
            [self.encoder_pars],
            self.encode(self.encoder_pars)
        )

    def compute_embedded_pars(self,
                              encoder_pars: np.ndarray) -> List:
        """
        Compute the inflated parameters
        """
        return self.compile_embedded_pars()(encoder_pars)

    def compile_inflate_pars(self) -> AFunction:
        """
        Compile a theano function that computes the inflated parameters
        """
        return aesara.function(
            [self.encoder_pars],
            self.encode_params(self.encoder_pars)
        )

    def compute_inflated_pars(self,
                              encoder_pars: np.ndarray) -> List:
        """
        Compute the inflated parameters
        """
        return self.compile_inflate_pars()(encoder_pars)
