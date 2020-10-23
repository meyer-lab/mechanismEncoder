"""
Materials for a simple linear encoder, and its analytical reverse.
"""

import theano.tensor as T

class dA:
    """A simple linear autoencoder. """

    def __init__(self, n_visible=50, n_hidden=10, n_params=12):
        assert n_hidden < n_visible
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_params = n_params
        self.encoder_pars = T.specify_shape(T.vector('W'),
                                            (n_visible * n_hidden,))
        self.W = T.reshape(self.encoder_pars, (n_visible, n_hidden))
        self.inflate_pars = T.specify_shape(T.vector('W_p'),
                                            (n_hidden * n_params,))
        self.W_p = T.reshape(self.inflate_pars, (n_hidden, n_params))

    def encode(self, input_data):
        """ Run the input through the encoder. """
        return T.dot(input_data, self.W)

    def encode_params(self, input_data):
        """ Run the encoder and then inflate to parameters. """
        return T.dot(self.encode(input_data), self.W_p)
    
    def decode(self, embedded_data):
        """ Run the input through the analytical decoder. """
        return T.dot(embedded_data, T.nlinalg.pinv(self.W))
