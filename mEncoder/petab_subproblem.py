import os
import petab

import pandas as pd
import numpy as np

from amici.petab_import import PysbPetabProblem
from pypesto.petab.pysb_importer import PetabImporterPysb

from . import parameter_boundaries_scales, MODEL_FEATURE_PREFIX, \
    load_pathway, basedir

from typing import Tuple


def load_petab(datafiles: Tuple[str, str],
               pathway_name: str,
               par_input_scale: float):
    """
    Imports data from a csv and converts it to the petab format. This
    function is used to connect the mechanistic model to the specified data
    in order to defines the loss function of the autoencoder up to the
    inflated parameters

    :param datafiles:
        tuple of paths to measurements and conditions files

    :param pathway_name:
        name of pathway to use for model

    :param par_input_scale:
        absolute value of upper/lower bounds for input parameters in log10
        scale, also influence l2 regularization strength (std of gaussian
        prior is par_input_scale/2)
    """
    measurement_table = pd.read_csv(datafiles[0], index_col=0, sep='\t')
    condition_table = pd.read_csv(datafiles[1], index_col=0, sep='\t')

    model = load_pathway(pathway_name)

    features = [par for par in model.parameters
                if par.name.startswith(MODEL_FEATURE_PREFIX)]

    # CONDITION TABLE
    # this defines the different samples. here we define the mapping from
    # input parameters to model parameters
    for feature in features:
        condition_table[feature.name] = [
            f'{feature.name}_{s}' for s in condition_table.index
        ]

    # MEASUREMENT TABLE
    # filter for whats available in the model:
    measurement_table = measurement_table.loc[
        measurement_table[petab.OBSERVABLE_ID].apply(
            lambda x: x in [expr.name for expr in model.expressions]
        ), :
    ]

    # OBSERVABLE TABLE
    # this defines how model simulation are linked to experimental data,
    # currently this uses quantities that were already defined in the model
    observable_ids = measurement_table[petab.OBSERVABLE_ID].unique()

    observable_table = pd.DataFrame({
        petab.OBSERVABLE_ID: observable_ids,
        petab.OBSERVABLE_NAME: observable_ids,
        petab.OBSERVABLE_FORMULA: ['0.0' for _ in observable_ids],
    }).set_index(petab.OBSERVABLE_ID)
    observable_table[petab.NOISE_DISTRIBUTION] = 'normal'
    observable_table[petab.NOISE_FORMULA] = '1.0'

    # PARAMETER TABLE
    # this defines the full set of parameters including boundaries, nominal
    # values, scale, priors and whether they will be estimated or not.
    params = [par for par in model.parameters
              if not par.name.startswith(MODEL_FEATURE_PREFIX)]

    transforms = {
        'lin': lambda x: x,
        'log10': lambda x: np.power(10.0, x)
    }

    # base definition of id, upper and lower bounds, scale and value
    param_defs = [{
        petab.PARAMETER_ID: par.name,
        petab.LOWER_BOUND: transforms[parameter_boundaries_scales[
            par.name.split('_')[-1]][2]
        ](parameter_boundaries_scales[par.name.split('_')[-1]][0]),
        petab.UPPER_BOUND: transforms[parameter_boundaries_scales[
            par.name.split('_')[-1]][2]
        ](parameter_boundaries_scales[par.name.split('_')[-1]][1]),
        petab.PARAMETER_SCALE: parameter_boundaries_scales[
            par.name.split('_')[-1]][2],
        petab.NOMINAL_VALUE: par.value,
    } for par in params]

    # add additional input parameters for every sample
    for cond in condition_table.index.values:
        param_defs.extend([{
            petab.PARAMETER_ID: f'{par.name}_{cond}',
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
        petab.PARAMETER_SCALE_NORMAL if name.startswith('INPUT')
        else petab.PARAMETER_SCALE_UNIFORM
        for name in parameter_table.index
    ]
    parameter_table[petab.OBJECTIVE_PRIOR_PARAMETERS] = [
        f'0.0;{par_input_scale * 2}' if name.startswith('INPUT')
        else f'{parameter_table.loc[name, petab.LOWER_BOUND]};'
             f'{parameter_table.loc[name, petab.UPPER_BOUND]}'
        for name in parameter_table.index
    ]

    return PetabImporterPysb(PysbPetabProblem(
        measurement_df=measurement_table,
        condition_df=condition_table,
        observable_df=observable_table,
        parameter_df=parameter_table,
        pysb_model=model,
    ), output_folder=os.path.join(
        basedir, 'amici_models',
        f'{model.name}_'
        f'{os.path.splitext(os.path.basename(datafiles[0]))[0]}_petab'
    ))
