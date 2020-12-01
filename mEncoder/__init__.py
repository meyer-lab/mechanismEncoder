import os
import amici.pysb_import
import logging
import importlib
import sys
import amici
import pysb.export
import matplotlib.pyplot as plt
from typing import Tuple

basedir = os.path.dirname(os.path.dirname(__file__))
figures_path = os.path.join(basedir, 'figures')


def load_pathway(pathway_name: str) -> pysb.Model:
    model_file = os.path.join(basedir, 'pathways', pathway_name + '.py')
    return amici.pysb_import.pysb_model_from_path(model_file)


def load_model(pathway_name: str,
               force_compile: bool = True) -> Tuple[amici.AmiciModel,
                                                    amici.AmiciSolver]:

    model = load_pathway(pathway_name)
    outdir = os.path.join(basedir, 'amici_models', model.name)
    with open(os.path.join(basedir, 'pysb_models',
                           model.name + '.py'), 'w') as file:
        file.write(pysb.export.export(model, 'pysb_flat'))

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

    return amici_model, amici_model.getSolver()


def plot_and_save_fig(filename):
    plt.tight_layout()
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)

    if filename is not None:
        plt.savefig(os.path.join(figures_path, filename))


parameter_boundaries_scales = {
    'kdeg': (-3, -1, 'log10'),      # [1/[t]]
    'eq': (1, 2, 'log10'),          # [[c]]
    'bias': (-1, 1, 'lin'),         # [-]
    'kcat': (1, 3, 'log10'),        # [1/([t]*[c])]
    'scale': (-2, 2, 'log10'),       # [1/[c]]
    'offset': (-2, 2, 'log10'),     # [[c]]
    'weight': (-1, 1, 'lin'),       # [-]
}

MODEL_FEATURE_PREFIX = 'INPUT_'