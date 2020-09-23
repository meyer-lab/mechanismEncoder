""" Testing file for the linear encoder. """

import numpy as np
import theano
import theano.tensor as T
from ..encoder import dA


def test_encoder():
    """
    Test that the linear encoder at least runs.
    """
    enc = dA()
    unk = np.ones(enc.n_visible)

    a = T.dvector("tempVar")
    fexpr = enc.encode(a)

    # Calculate the Jacobian
    J = theano.scan(lambda i, y, x: T.grad(fexpr[i], a), sequences=T.arange(fexpr.shape[0]), non_sequences=[fexpr, a])[0]

    f = theano.function([a], fexpr)
    fprime = theano.function([a], J)

    output = f(unk)
    Doutput = fprime(unk)

    assert output.size == enc.n_hidden
    np.testing.assert_allclose(Doutput.T, enc.W.get_value())
