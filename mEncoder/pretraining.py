from pypesto import Problem, Result
from amici.petab_import import PysbPetabProblem
from pypesto.petab.pysb_importer import PetabImporterPysb
from pypesto.optimize import OptimizeOptions, minimize
from pypesto.store import OptimizationResultHDF5Writer
from pypesto.visualize import waterfall, parameters
from pypesto.objective.aesara import AesaraObjective

import petab
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from pysb import Model

from .autoencoder import MechanisticAutoEncoder
from . import (
    MODEL_FEATURE_PREFIX, load_pathway,
)

basedir = os.path.dirname(os.path.dirname(__file__))


def generate_per_sample_pretraining_problems(
        ae: MechanisticAutoEncoder,
        sample: str
) -> PetabImporterPysb:
    """
    Creates a pypesto problem that can be used to train the
    mechanistic model individually on every sample

    :param ae:
        Mechanistic autoencoder that will be pretrained

    :returns:
        Dict of pypesto problems. Keys are sample names.
    """
    # construct problem based on petab for pypesto subproblem
    ae.petab_importer.petab_problem.parameter_df['estimate'] = [
        not x.startswith(MODEL_FEATURE_PREFIX) and
        ae.petab_importer.petab_problem.parameter_df['estimate'][x]
        for x in ae.petab_importer.petab_problem.parameter_df.index
    ]
    pp = ae.petab_importer.petab_problem
    pp.parameter_df[petab.LOWER_BOUND] /= 10 ** ae.par_modulation_scale
    pp.parameter_df[petab.UPPER_BOUND] *= 10 ** ae.par_modulation_scale
    # create fresh model from scratch since the petab imported one already
    # has the observables added and this might lead to issues.
    clean_model = load_pathway('pw_' + ae.pathway_name)

    mdf = pp.measurement_df[
        pp.measurement_df[petab.PREEQUILIBRATION_CONDITION_ID]
        == sample
    ]
    cdf = pp.condition_df[[
        name.startswith(sample)
        for name in pp.condition_df.index
    ]]
    vdf = pd.read_csv(
        os.path.join('data',
                     f'{ae.data_name}__'
                     f'{sample}__visualization.tsv'), sep='\t'
    )
    spars = set(
        e
        for t in mdf[petab.OBSERVABLE_PARAMETERS].apply(lambda x: x.split(';'))
        for e in t
    )
    pdf = pp.parameter_df[[
        (not name.startswith(MODEL_FEATURE_PREFIX) and (
            (not name.endswith('_scale') and not name.endswith('offset'))
            or name in spars
         ))
        or name.endswith(sample)
        for name in pp.parameter_df.index
    ]]

    return PetabImporterPysb(PysbPetabProblem(
        parameter_df=pdf,
        observable_df=pp.observable_df,
        measurement_df=mdf,
        condition_df=cdf,
        visualization_df=vdf,
        pysb_model=Model(base=clean_model, name=pp.pysb_model.name),
    ), output_folder=os.path.join(
        basedir, 'amici_models',
        f'{pp.pysb_model.name}_'
        f'{ae.data_name}_petab'
    ))


def generate_cross_sample_pretraining_problem(
        ae: MechanisticAutoEncoder
) -> Problem:
    """
    Creates a pypesto problem that can be used to train population
    parameters as well as individual sample specific parameters. This is
    effectively just the unconstrained petab subproblem.

    :param ae:
        Mechanistic autoencoder that will be pretrained

    :returns:
        pypesto Problem
    """

    obj = AesaraObjective(
        ae.pypesto_subproblem.objective, ae.x_embedding,
        ae.embedding_model_pars
    )

    x_names = ae.x_names[ae.n_encode_weights:]

    return Problem(
        objective=obj,
        x_names=x_names,
        lb=[-np.inf for _ in x_names],
        ub=[np.inf for _ in x_names],
    )


def pretrain(problem: Problem, startpoint_method: Callable, nstarts: int,
             optimizer, engine=None) -> Result:
    """
    Pretrain the provided problem via optimization.

    :param problem:
        problem that defines the pretraining optimization problem

    :param startpoint_method:
        function that generates the initial points for optimization. In most
        cases this uses results from previous pretraining steps.

    :param nstarts:
        number of local optimizations to perform

    :param fatol:
        absolute function tolerance for termination of optimization

    :param subspace:
        fides subspace to use, fides.SubSpaceDim.FULL becomes quite slow for
        for anything with over 1k parameters

    :param maxiter:
        maximum number of iterations
    """

    optimize_options = OptimizeOptions(
        startpoint_resample=True, allow_failed_starts=False,
    )

    return minimize(
        problem, optimizer, n_starts=nstarts, options=optimize_options,
        startpoint_method=startpoint_method,
        engine=engine
    )


def store_and_plot_pretraining(result: Result, pretraindir: str, prefix: str):
    """
    Store optimziation results in HDF5 as well as csv for later reuse. Also
    saves some visualization for debugging purposes.

    :param result:
        result from pretraining

    :param pretraindir:
        directory in which results and plots will be stored

    :param prefix:
        prefix for file names that can be used to differentiate between
        different pretraining stages as well as models/datasets.
    """
    # store full results as hdf5
    rfile = os.path.join(pretraindir, prefix + '.hdf5')
    if os.path.exists(rfile):
        # temp bugfix for https://github.com/ICB-DCM/pyPESTO/issues/529
        os.remove(rfile)
    writer = OptimizationResultHDF5Writer(rfile)
    writer.write(result, overwrite=True)

    # store parameter values, this will be used in subsequent steps
    parameter_df = pd.DataFrame(
        [r for r in result.optimize_result.get_for_key('x')
         if r is not None],
        columns=result.problem.x_names
    )
    parameter_df.to_csv(os.path.join(pretraindir, prefix + '.csv'))

    # do plotting
    waterfall(result, scale_y='log10', offset_y=0.0)
    plt.tight_layout()
    plt.savefig(os.path.join(pretraindir, prefix + '_waterfall.pdf'))

    if result.problem.dim_full < 2e3:
        parameters(result)
        plt.tight_layout()
        plt.savefig(os.path.join(pretraindir, prefix + '_parameters.pdf'))
