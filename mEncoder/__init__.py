import os
import amici.pysb_import
import logging
import importlib
import sys
import amici
import pysb.export
from typing import Tuple


def load_model(force_compile: bool = True) -> Tuple[amici.AmiciModel,
                                                    amici.AmiciSolver]:
    from .pathway_FLT3_MAPK_AKT_STAT import model
    basedir = os.path.dirname(os.path.dirname(__file__))
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
