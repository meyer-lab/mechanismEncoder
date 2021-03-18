import os
import amici.pysb_import
import logging
import importlib
import sys
import re
import pysb
import sympy as sp
import amici
import pysb.export
import matplotlib.pyplot as plt
from typing import Tuple

basedir = os.path.dirname(os.path.dirname(__file__))
figures_path = os.path.join(basedir, 'figures')


def load_pathway(pathway_name: str) -> pysb.Model:
    model_file = os.path.join(basedir, 'pathways', pathway_name + '.py')

    model = amici.pysb_import.pysb_model_from_path(model_file)

    with open(os.path.join(basedir, 'pysb_models',
                           model.name + '.py'), 'w') as file:
        file.write(pysb.export.export(model, 'pysb_flat'))

    return model


def load_model(pathway_name: str,
               force_compile: bool = True,
               add_observables: bool = False) -> Tuple[amici.AmiciModel,
                                                       amici.AmiciSolver]:

    model = load_pathway(pathway_name)
    outdir = os.path.join(basedir, 'amici_models', model.name)

    # extend observables
    if add_observables:
        for obs in model.observables:
            if re.match(r'[p|t][A-Z0-9]+[SYT0-9_]*', obs.name):
                offset = pysb.Parameter(obs.name + '_offset', 0.0)
                scale = pysb.Parameter(obs.name + '_scale', 1.0)
                pysb.Expression(obs.name + '_obs',
                                sp.log(scale * (obs + offset)))

    if force_compile or not os.path.exists(os.path.join(outdir, model.name,
                                                        model.name + '.py')):
        os.makedirs(outdir, exist_ok=True)
        amici.pysb_import.pysb2amici(model,
                                     outdir,
                                     verbose=logging.DEBUG,
                                     observables=[
                                         expr.name
                                         for expr in model.expressions
                                         if expr.name.endswith('_obs')
                                     ],
                                     constant_parameters=[])

    sys.path.insert(0, os.path.abspath(outdir))
    model_module = importlib.import_module(model.name)

    amici_model = model_module.getModel()
    solver = amici_model.getSolver()

    solver.setMaxSteps(int(1e5))
    solver.setAbsoluteToleranceSteadyState(1e-2)
    solver.setRelativeToleranceSteadyState(1e-8)

    return amici_model, solver


def plot_and_save_fig(filename):
    plt.tight_layout()
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)

    if filename is not None:
        plt.savefig(os.path.join(figures_path, filename))


parameter_boundaries_scales = {
    'kdeg': (-8, -3, 'log10'),      # [1/[t]]
    'eq': (-4, 4, 'log10'),          # [[c]]
    'kcat': (-4, 4, 'log10'),        # [1/([t]*[c])]
    'scale': (-4, 0, 'log10'),       # [1/[c]]
    'offset': (0, 4, 'log10'),     # [[c]]
    'weight': (-1, 1, 'lin'),       # [-]
    'koff': (-5, 2, 'log10'),      # [1/[t]]
    'kd':   (-4, 4, 'log10'),       # [[c]]
}

MODEL_FEATURE_PREFIX = 'INPUT_'
