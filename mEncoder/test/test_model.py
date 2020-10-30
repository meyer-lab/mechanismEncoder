from .. import load_model
from ..autoencoder import load_petab, MechanisticAutoEncoder
from ..generate_data import generate_synthetic_data

import amici
import petab
import itertools as itt

import numpy as np

pathway_model = 'FLT3_MAPK_AKT_STAT'


def test_model_compilation():
    """
    Test that we can load and simulate the mechanistic model in AMICI
    """
    model, solver = load_model('pw_' + pathway_model)
    amici.runAmiciSimulation(model, solver)


def test_petab_loading():
    """
    Test that we can load the mechanistic model plus data in PEtab
    """
    datafile = generate_synthetic_data(pathway_model)
    petab_importer = load_petab(datafile, 'pw_' + pathway_model)
    petab.lint.lint_problem(petab_importer.petab_problem)


def test_pypesto_objective():
    """
    Test that we can load and evaluate the theano loss function and gradient
    for the full autoencoder model, checks accuracy of gradient via finite
    differences
    """
    datafile = generate_synthetic_data(pathway_model)
    n_hidden = 10

    mae = MechanisticAutoEncoder(n_hidden, datafile, pathway_model)
    objective = mae.generate_pypesto_objective()
    x = np.random.random((mae.n_encoder_pars + mae.n_kin_params,))
    x[0:mae.n_encoder_pars] /= 10
    assert np.isfinite(objective.get_fval(x))
    assert not any(np.isnan(objective.get_grad(x)))
    fd_df = objective.check_grad(
        x, eps=1e-3,
        x_indices=list(itt.chain(range(5), range(mae.n_encoder_pars,
                                                 mae.n_encoder_pars+5)))
    )
    assert (fd_df['abs_err'] < 1e-2).all()


def test_pypesto_optimization():
    """
    Test that we can minimize the loss using pypesto
    """
    datafile = generate_synthetic_data(pathway_model)
    n_hidden = 10

    mae = MechanisticAutoEncoder(n_hidden, datafile, pathway_model)
    mae.train(maxiter=5)
