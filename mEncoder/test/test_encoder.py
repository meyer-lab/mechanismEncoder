""" Testing file for the linear encoder. """

import numpy as np
import theano
import theano.tensor as tt
from ..encoder import AutoEncoder


def test_encoder():
    """
    Test that the linear encoder at least runs.
    """
    data = np.random.rand(100, 10)
    enc = AutoEncoder(data)
    unk = np.ones(enc.n_encoder_pars)

    a = tt.dvector("tempVar")
    fexpr = enc.encode(a)

    f = theano.function([a], fexpr)
    output = f(unk)

    assert output.shape == (data.shape[0], enc.n_hidden)


def test_init():
    """
    Test that we can initialize W by SVD of the data.
    """
    data = np.random.rand(100, 10)
    enc = AutoEncoder(data, n_hidden=3)

    a = tt.dvector("tempVar")
    fexpr = tt.nlinalg.norm(enc.decode(enc.encode(a), a) - enc.data, ord=None)
    fprime = tt.grad(fexpr, a)
    f = theano.function([a], fprime)

    unk = enc.initialW()
    output = f(unk)

    # This should be the minimum of a linear encoder
    np.testing.assert_allclose(output, 0.0, atol=1e-9)


def test_reg():
    """
    Test that regularization works for a test problem.
    """
    data = np.random.rand(100, 10)
    enc = AutoEncoder(data, n_hidden=3)

    a = tt.dvector("tempVar")
    costf = enc.regularize(a, l2=0.01, ortho=0.0001)
    f = theano.function([a], costf)

    unk = enc.initialW()
    output = f(unk)

    assert output > 0.0
