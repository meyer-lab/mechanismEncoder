from .. import load_model
from ..autoencoder import load_petab, MechanisticAutoEncoder
from ..generate_data import generate_synthetic_data
from .. import MODEL_FEATURE_PREFIX

import amici
import petab

import numpy as np

pathway_model = 'pw_FLT3_MAPK_AKT_STAT'


def test_model_compilation():
    """
    Test that we can load and simulate the mechanistic model in AMICI
    """
    model, solver = load_model(pathway_model)
    amici.runAmiciSimulation(model, solver)


def test_petab_loading():
    """
    Test that we can load the mechanistic model plus data in PEtab
    """
    datafile = generate_synthetic_data(pathway_model)
    petab_importer = load_petab(datafile, pathway_model)
    petab.lint.lint_problem(petab_importer.petab_problem)


def test_theano_objective():
    """
    Test that we can load the theano objective for the mechanistic model
    """
    datafile = generate_synthetic_data(pathway_model)
    n_hidden = 10

    mae = MechanisticAutoEncoder(n_hidden, datafile, pathway_model)
    loss = mae.compile_loss()
    loss(np.random.random((mae.n_encoder_pars,)),
         np.random.random((mae.n_inflate_pars,)),
         np.random.random((mae.n_kin_params,)),)
