import os
import amici.pysb_import
import logging
import re
import pysb
import sys
import sympy as sp
import amici
import pypesto
import pysb.export
import matplotlib.pyplot as plt
from typing import Tuple

basedir = os.path.dirname(os.path.dirname(__file__))
figures_path = os.path.join(basedir, 'figures')


def load_pathway(pathway_name: str) -> pysb.Model:
    model_file = os.path.join(basedir, 'pathways', pathway_name + '.py')
    sys.path.insert(0, os.path.join(basedir, 'pathways'))
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

    model_module = amici.import_model_module(model.name, outdir)

    amici_model = model_module.getModel()
    solver = amici_model.getSolver()

    apply_solver_settings(solver)

    return amici_model, solver


def plot_and_save_fig(filename):
    plt.tight_layout()
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)

    if filename is not None:
        plt.savefig(os.path.join(figures_path, filename))


def apply_solver_settings(solver):
    solver.setMaxSteps(int(1e5))
    solver.setAbsoluteTolerance(1e-12)
    solver.setRelativeTolerance(1e-12)
    solver.setAbsoluteToleranceSteadyState(1e-8)
    solver.setRelativeToleranceSteadyState(1e-8)


def apply_objective_settings(problem):
    if isinstance(problem.objective, pypesto.objective.AmiciObjective):
        amiobjective = problem.objective
    elif isinstance(problem.objective, pypesto.objective.AggregatedObjective):
        amiobjective = problem.objective._objectives[0]
    elif isinstance(problem.objective,
                    pypesto.objective.aesara.AesaraObjective):
        base_objective = problem.objective.base_objective
        if isinstance(base_objective, pypesto.objective.AggregatedObjective):
            amiobjective = base_objective._objectives[0]
        elif isinstance(base_objective, pypesto.objective.AmiciObjective):
            amiobjective = base_objective

    amiobjective.guess_steadystate = False
    apply_solver_settings(amiobjective.amici_solver)
    for e in amiobjective.edatas:
        e.reinitializeFixedParameterInitialStates = True


parameter_boundaries_scales = {
    'kdeg': (-6, -2, 'log10'),       # [1/[t]]
    'eq': (-4, 4, 'log10'),          # [[c]]
    'kcat': (-3, 3, 'log10'),        # [1/([t]*[c])]
    'kr': (-3, 3, 'log10'),          # [-]
    'scale': (0, 10, 'lin'),         # [1/[c]]
    'offset': (-10, 10, 'lin'),        # [[c]]
    'weight': (-1, 1, 'lin'),        # [-]
    'koff': (-3, 2, 'log10'),        # [1/[t]]
    'kd':   (-3, 2, 'log10'),       # [[c]]
    'kw':   (-3, 2, 'log10'),        # [1/[c]]
}

MODEL_FEATURE_PREFIX = 'INPUT_'
