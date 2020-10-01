import amici.pysb_import
import logging

import os


def test_model_compilation():
    from ..pathway_FLT3_MAPK_AKT_STAT import model
    basedir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    outdir = os.path.join(basedir, 'amici_models')
    os.makedirs(outdir, exist_ok=True)
    amici.pysb_import.pysb2amici(model,
                                 outdir,
                                 verbose=logging.DEBUG,
                                 observables=[],
                                 constant_parameters=[])
