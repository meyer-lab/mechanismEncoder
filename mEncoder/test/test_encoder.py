""" Testing file for the linear encoder. """

import numpy as np
import theano
import theano.tensor as T
from ..encoder import AutoEncoder


def test_encoder():
    """
    Test that the linear encoder at least runs.
    """
    data = np.random.rand(100, 10)
    enc = AutoEncoder(data)
    unk = np.ones(enc.n_encoder_pars)

    a = T.dvector("tempVar")
    fexpr = enc.encode(a)

    f = theano.function([a], fexpr)
    output = f(unk)

    assert output.shape == (data.shape[0], enc.n_hidden)
