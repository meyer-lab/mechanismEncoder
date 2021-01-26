from pypesto import Problem, Objective, Result
from amici.petab_import import PysbPetabProblem
from pypesto.petab.pysb_importer import PetabImporterPysb
from pypesto.optimize import FidesOptimizer, OptimizeOptions, minimize
from pypesto.store import OptimizationResultHDF5Writer
from pypesto.visualize import waterfall, parameters

import petab
import os
import fides
import pandas as pd
import numpy as np
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt

from typing import Dict, Callable
from pysb import Model

from .autoencoder import MechanisticAutoEncoder
from . import parameter_boundaries_scales, MODEL_FEATURE_PREFIX, load_pathway

basedir = os.path.dirname(os.path.dirname(__file__))


def generate_per_sample_pretraining_problems(
        ae: MechanisticAutoEncoder
) -> Dict[str, Problem]:
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

    samples = [
        c for c in pp.condition_df.index
        if c in pp.measurement_df[petab.PREEQUILIBRATION_CONDITION_ID].unique()
    ]

    return {
        sample: PetabImporterPysb(PysbPetabProblem(
            parameter_df=pp.parameter_df[[
                not name.startswith(MODEL_FEATURE_PREFIX)
                or name.endswith(sample)
                for name in pp.parameter_df.index
            ]],
            observable_df=pp.observable_df,
            measurement_df=pp.measurement_df[
                pp.measurement_df[petab.PREEQUILIBRATION_CONDITION_ID]
                == sample
            ],
            condition_df=pp.condition_df[[
                name.startswith(sample)
                for name in pp.condition_df.index
            ]],
            pysb_model=Model(base=clean_model, name=pp.pysb_model.name),
        ), output_folder=os.path.join(
            basedir, 'amici_models',
            f'{pp.pysb_model.name}_'
            f'{ae.data_name}_petab'
        )).create_problem()
        for sample in samples
    }


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
    problem = ae.petab_importer.create_problem()
    # make unbounded
    problem.ub_full[:] = np.inf
    problem.lb_full[:] = -np.inf
    return problem


def generate_encoder_inflate_pretraining_problem(
        ae: MechanisticAutoEncoder, pretrained_inputs: pd.DataFrame,
        pars: pd.DataFrame
) -> Problem:
    """
    Creates a pypesto problem that can be used to train encoder and inflate
    parameters. This is done based on the precomputed input parameters that
    were generated during cross sample pretraining. This function defines a
    least squares problem ||W_p*W*x - p||, where `W` is the encoder matrix,
    `W_p` is the inflate matrix, x is the input data and p are the
    pretrained input parameters. Optimization is performed over variables
    `W_p` and `W`.

    :param ae:
        Mechanistic autoencoder that will be pretrained

    :param pretrained_inputs:
        pretrained input parameters computed by performing cross sample
        pretraining

    :param pars:
        corresponding population input parameters that were pretrained along
        with the pretrained inputs. This input does not affect the solution,
        but will be stored as fixed parameters in the result such that it is
        available in later pretraining steps

    :returns:
        pypesto Problem
    """
    least_squares = .5*tt.sum(tt.power(
        ae.encode_params(ae.encoder_pars) -
        pretrained_inputs[ae.sample_names].values.T, 2
    )[:])

    loss = theano.function([ae.encoder_pars], least_squares)
    loss_grad = theano.function(
        [ae.encoder_pars], theano.grad(least_squares, [ae.encoder_pars])
    )

    return Problem(
        objective=Objective(
            fun=lambda x: np.float(loss(x[:ae.n_encoder_pars])),
            grad=lambda x: loss_grad(x[:ae.n_encoder_pars])[0]
        ),
        ub=[np.inf for _ in ae.x_names[:ae.n_encoder_pars]],
        lb=[-np.inf for _ in ae.x_names[:ae.n_encoder_pars]],
        lb_init=[parameter_boundaries_scales[name.split('_')[-1]][0]
                 for name in ae.x_names[:ae.n_encoder_pars]],
        ub_init=[parameter_boundaries_scales[name.split('_')[-1]][1]
                 for name in ae.x_names[:ae.n_encoder_pars]],
        x_names=ae.x_names[:ae.n_encoder_pars] + list(pars.index),
        x_fixed_indices=list(range(ae.n_encoder_pars,
                                   ae.n_encoder_pars+ae.n_kin_params)),
        dim_full=ae.n_encoder_pars+ae.n_kin_params,
        x_fixed_vals=pars.values
    )


def pretrain(problem: Problem, startpoint_method: Callable, nstarts: int,
             fatol: float = 1e-2,
             subspace: fides.SubSpaceDim = fides.SubSpaceDim.FULL,
             maxiter: int = int(1e3)):
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
    opt = FidesOptimizer(
        hessian_update=fides.BFGS(),
        options={
            'maxtime': 3600,
            fides.Options.FATOL: fatol,
            fides.Options.MAXTIME: 7200,
            fides.Options.MAXITER: maxiter,
            fides.Options.SUBSPACE_DIM: subspace,
            fides.Options.REFINE_STEPBACK: False,
            fides.Options.STEPBACK_STRAT: fides.StepBackStrategy.SINGLE_REFLECT
        }
    )

    problem.objective._objectives[0].amici_solver.setMaxSteps(int(1e5))

    optimize_options = OptimizeOptions(
        startpoint_resample=True, allow_failed_starts=True,
    )

    return minimize(
        problem, opt,
        n_starts=nstarts, options=optimize_options,
        startpoint_method=startpoint_method,
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
