import os
import petab

import pandas as pd
import numpy as np

from amici.petab_import import PysbPetabProblem
from pypesto.petab.pysb_importer import PetabImporterPysb

from . import parameter_boundaries_scales, MODEL_FEATURE_PREFIX, \
    load_pathway, basedir

from typing import Tuple, Sequence


def load_petab(datafiles: Tuple[str, str, str],
               pathway_name: str,
               par_input_scale: float,
               samples: Sequence[str] = None):
    """
    Imports data from a csv and converts it to the petab format. This
    function is used to connect the mechanistic model to the specified data
    in order to defines the loss function of the autoencoder up to the
    inflated parameters

    :param datafiles:
        tuple of paths to measurements, conditions and observables files

    :param pathway_name:
        name of pathway to use for model

    :param par_input_scale:
        absolute value of upper/lower bounds for input parameters in log10
        scale, also influence l2 regularization strength (std of gaussian
        prior is par_input_scale/2)
    """
    measurement_table = pd.read_csv(datafiles[0], index_col=0, sep='\t')
    condition_table = pd.read_csv(datafiles[1], index_col=0, sep='\t')
    observable_table = pd.read_csv(datafiles[2], index_col=0, sep='\t')

    if samples:
        measurement_table = measurement_table[
            measurement_table[petab.PREEQUILIBRATION_CONDITION_ID].apply(
                lambda x: x in samples
            )
        ]
        condition_table = condition_table.loc[
            [c for c in condition_table.index
             if c.split('__')[0] in samples], :
        ]

    model = load_pathway(pathway_name)

    features = [par for par in model.parameters
                if par.name.startswith(MODEL_FEATURE_PREFIX)]

    # CONDITION TABLE
    # this defines the different samples. here we define the mapping from
    # input parameters to model parameters

    preeq_conds = dict()
    for cond in list(condition_table.index):
        candidates = measurement_table[measurement_table[
            petab.SIMULATION_CONDITION_ID] == cond
        ][petab.PREEQUILIBRATION_CONDITION_ID].unique()
        if len(candidates) > 1:
            raise RuntimeError('Found multiple different preequilibration '
                               f'conditions {candidates} for condition '
                               f'{cond}, which is not supported.')
        if len(candidates) == 0:
            condition_table.drop(index=cond, inplace=True)
            continue
        preeq_conds[cond] = candidates[0]

    for feature in features:
        condition_table[feature.name] = [
            f'{feature.name}__{preeq_conds[s]}' for s in condition_table.index
        ]

    # PARAMETER TABLE
    # this defines the full set of parameters including boundaries, nominal
    # values, scale, priors and whether they will be estimated or not.

    obs_pars = sorted(list(set([
        par
        for pars in measurement_table[petab.OBSERVABLE_PARAMETERS]
        for par in pars.split(';')
        if par and any(
            obs in par for obs in observable_table.index
        )
    ])))

    params = [par.name for par in model.parameters
              if par.name not in condition_table.columns] + obs_pars

    transforms = {
        'lin': lambda x: x,
        'log10': lambda x: np.power(10.0, x)
    }

    # base definition of id, upper and lower bounds, scale and value
    param_defs = [{
        petab.PARAMETER_ID: par,
        petab.LOWER_BOUND: transforms[parameter_boundaries_scales[
            par.split('_')[-1]][2]
        ](parameter_boundaries_scales[par.split('_')[-1]][0]),
        petab.UPPER_BOUND: transforms[parameter_boundaries_scales[
            par.split('_')[-1]][2]
        ](parameter_boundaries_scales[par.split('_')[-1]][1]),
        petab.PARAMETER_SCALE: parameter_boundaries_scales[
            par.split('_')[-1]][2],
        petab.NOMINAL_VALUE: model.parameters[par].value
        if par in model.parameters.keys()
        else 4.0 if par.endswith('offset') and par.startswith('p')
        else 0.0 if par.endswith('offset') and par.startswith('t')
        else 1.0
    } for par in params]

    # add additional input parameters for every base condition
    for cond in measurement_table[
        petab.PREEQUILIBRATION_CONDITION_ID
    ].unique():
        param_defs.extend([{
            petab.PARAMETER_ID: f'{par.name}__{cond}',
            petab.LOWER_BOUND: 10**-par_input_scale,
            petab.UPPER_BOUND: 10**par_input_scale,
            petab.PARAMETER_SCALE: 'log10',
            petab.NOMINAL_VALUE: 1.0,
        } for par in features])

    # piece of codes allows disabling estimation for parameter by setting
    # equal upper and lower bounds, primarily for debugging purposes
    parameter_table = pd.DataFrame(param_defs).set_index(petab.PARAMETER_ID)
    parameter_table[petab.ESTIMATE] = (
        parameter_table[petab.LOWER_BOUND] !=
        parameter_table[petab.UPPER_BOUND]
    ).apply(lambda x: int(x))

    # add l2 regularization to input parameters
    parameter_table[petab.OBJECTIVE_PRIOR_TYPE] = [
        petab.PARAMETER_SCALE_NORMAL if name.startswith(MODEL_FEATURE_PREFIX)
        else petab.PARAMETER_SCALE_UNIFORM
        for name in parameter_table.index
    ]
    parameter_table[petab.OBJECTIVE_PRIOR_PARAMETERS] = [
        f'0.0;{par_input_scale}' if name.startswith(MODEL_FEATURE_PREFIX)
        else f'{parameter_table.loc[name, petab.LOWER_BOUND]};'
             f'{parameter_table.loc[name, petab.UPPER_BOUND]}'
        for name in parameter_table.index
    ]

    data_name = '__'.join(os.path.splitext(
        os.path.basename(datafiles[0])
    )[0].split('__')[:-1])

    return PetabImporterPysb(PysbPetabProblem(
        measurement_df=measurement_table,
        condition_df=condition_table,
        observable_df=observable_table,
        parameter_df=parameter_table,
        pysb_model=model,
    ), output_folder=os.path.join(
        basedir, 'amici_models',
        f'{model.name}_{data_name}_petab'
    ))


def filter_observables(petab_problem: petab.Problem):
    petab_problem.measurement_df = petab_problem.measurement_df.loc[
        petab_problem.measurement_df[petab.OBSERVABLE_ID].apply(
            lambda x: x in petab_problem.observable_df.index
        ), :
    ]
    # filter obsolete observable pars
    obs_pars = set(
        p
        for ir, r in petab_problem.measurement_df.iterrows()
        for p in r[petab.OBSERVABLE_PARAMETERS].split(';')
    )
    for par in list(petab_problem.parameter_df.index):
        if not par.endswith('_scale') and not par.endswith('_offset'):
            continue
        if par not in obs_pars:
            petab_problem.parameter_df.drop(index=par, inplace=True)

