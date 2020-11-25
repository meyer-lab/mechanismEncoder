from pypesto import Problem, Objective
from amici.petab_import import PysbPetabProblem
from pypesto.petab.pysb_importer import PetabImporterPysb
from pypesto.optimize import FidesOptimizer, OptimizeOptions, minimize

import petab
import os
import fides
import pandas as pd
import numpy as np
import theano
import theano.tensor as tt

from typing import Dict, Callable
from pysb import Model

from .autoencoder import MechanisticAutoEncoder
from .autoencoder import load_pathway
from . import parameter_boundaries_scales

basedir = os.path.dirname(os.path.dirname(__file__))


def generate_patient_sample_pretraining_problems(
        ae: MechanisticAutoEncoder
) -> Dict[str, Problem]:
    """
    Creates a pypesto objective function (this is the loss function) that
    needs to be minimized to train the respective autoencoder

    :param ae:
        Autoencoder that will be trained

    :returns:
        Objective function that needs to be minimized for training.
    """
    ae.petab_importer.petab_problem.parameter_df['estimate'] = [
        not x.startswith('INPUT') and
        ae.petab_importer.petab_problem.parameter_df['estimate'][x]
        for x in ae.petab_importer.petab_problem.parameter_df.index
    ]
    pp = ae.petab_importer.petab_problem
    pp.parameter_df[petab.LOWER_BOUND] /= 10 ** ae.par_modulation_scale
    pp.parameter_df[petab.UPPER_BOUND] *= 10 ** ae.par_modulation_scale
    # create fresh model from scratch since the petab imported one already
    # has the observables added and this might lead to issues.
    clean_model = load_pathway('pw_' + ae.pathway_name)

    return {
        cond: PetabImporterPysb(PysbPetabProblem(
            parameter_df=pp.parameter_df[[
                not name.startswith('INPUT_') or name.endswith(cond)
                for name in pp.parameter_df.index
            ]],
            observable_df=pp.observable_df,
            measurement_df=pp.measurement_df[
                pp.measurement_df[petab.SIMULATION_CONDITION_ID] == cond
                ],
            condition_df=pp.condition_df[[
                name == cond
                for name in pp.condition_df.index
            ]],
            pysb_model=Model(base=clean_model, name=pp.pysb_model.name),
        ), output_folder=os.path.join(
            basedir, 'amici_models',
            f'{pp.pysb_model.name}_'
            f'{ae.data_name}_petab'
        )).create_problem()
        for cond in pp.condition_df.index
    }


def generate_cross_sample_pretraining_problem(
        ae: MechanisticAutoEncoder
) -> Problem:
    """
    Creates a pypesto objective function (this is the loss function) that
    needs to be minimized to train the respective autoencoder

    :param ae:
        Autoencoder that will be trained

    :returns:
        Objective function that needs to be minimized for training.
    """
    pp = ae.petab_importer.petab_problem
    # add l2 regularization to input parameters
    pp.parameter_df[petab.OBJECTIVE_PRIOR_TYPE] = [
        petab.PARAMETER_SCALE_NORMAL if name.startswith('INPUT')
        else petab.PARAMETER_SCALE_UNIFORM
        for name in pp.parameter_df.index
    ]
    pp.parameter_df[petab.OBJECTIVE_PRIOR_PARAMETERS] = [
        f'0.0;{ae.par_modulation_scale*2}' if name.startswith('INPUT')
        else f'{pp.parameter_df.loc[name, petab.LOWER_BOUND]};'
             f'{pp.parameter_df.loc[name, petab.UPPER_BOUND]}'
        for name in pp.parameter_df.index
    ]

    return ae.petab_importer.create_problem()


def generate_encoder_inflate_pretraining_problem(
        ae: MechanisticAutoEncoder, pretrained_inputs: pd.DataFrame
) -> Problem:
    least_squares = .5*tt.sum(tt.power(
        ae.encode_params(ae.encoder_pars) -
        pretrained_inputs[ae.sample_names].values.T, 2
    )[:])

    loss = theano.function([ae.encoder_pars], least_squares)
    loss_grad = theano.function(
        [ae.encoder_pars], theano.grad(least_squares, [ae.encoder_pars])
    )

    return Problem(
        objective=Objective(fun=lambda x: np.float(loss(x)),
                            grad=lambda x: loss_grad(x)[0]),
        lb=[parameter_boundaries_scales[name.split('_')[-1]][0]
            for name in ae.x_names[:ae.n_encoder_pars]],
        ub=[parameter_boundaries_scales[name.split('_')[-1]][1]
            for name in ae.x_names[:ae.n_encoder_pars]],
        x_names=ae.x_names[:ae.n_encoder_pars]
    )


def pretrain(problem: Problem, startpoint_method: Callable, nstarts: int,
             fatol: float = 1e-2, unbounded=False):
    opt = FidesOptimizer(
        hessian_update=fides.BFGS(),
        options={
            'maxtime': 3600,
            fides.Options.FATOL: fatol,
            fides.Options.MAXTIME: 7200,
            fides.Options.MAXITER: 1e3,
            fides.Options.SUBSPACE_DIM: fides.SubSpaceDim.FULL,
            fides.Options.REFINE_STEPBACK: False,
            fides.Options.STEPBACK_STRAT: fides.StepBackStrategy.SINGLE_REFLECT

        }
    )

    optimize_options = OptimizeOptions(
        startpoint_resample=True, allow_failed_starts=True,
        unbounded_optimization=unbounded
    )

    return minimize(
        problem, opt,
        n_starts=nstarts, options=optimize_options,
        startpoint_method=startpoint_method,
    )
