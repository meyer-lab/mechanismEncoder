from .. import load_model
from ..model_objective import load_petab, load_theano
from ..generate_data import generate_synthetic_data

import amici
import petab


def test_model_compilation():
    """
    Test that we can load and simulate the mechanistic model in AMICI
    """
    model, solver = load_model()
    amici.runAmiciSimulation(model, solver)


def test_petab_loading():
    """
    Test that we can load the mechanistic model plus data in PEtab
    """
    datafile = generate_synthetic_data()
    petab_importer = load_petab(datafile)
    petab.lint.lint_problem(petab_importer.petab_problem)


def test_theano_objective():
    """
    Test that we can load the theano objective for the mechanistic model
    """
    datafile = generate_synthetic_data()
    objective = load_theano(datafile)
