from .. import load_model
from ..model_objective import load_petab, load_theano
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
    petab_importer = load_petab(datafile, pathway_model)
    n_hidden = 10
    n_visible = len(petab_importer.petab_problem.observable_df)
    n_samples = len(petab_importer.petab_problem.condition_df)
    n_kin_pars = sum(not x_id.startswith(MODEL_FEATURE_PREFIX)
                     for x_id in petab_importer.petab_problem.x_ids)
    n_enc_pars = n_hidden * n_visible
    n_model_inputs = int(sum(x_id.startswith(MODEL_FEATURE_PREFIX)
                             for x_id in
                             petab_importer.petab_problem.x_ids)/n_samples)
    n_defl_pars = n_hidden * n_model_inputs

    loss = load_theano(datafile, pathway_model, n_hidden)
    loss(np.random.random((n_enc_pars,)),
         np.random.random((n_defl_pars,)),
         np.random.random((n_kin_pars,)),)
