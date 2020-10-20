import os
import re

import pandas as pd
import numpy as np

import petab
from amici.petab_import import PysbPetabProblem
from pypesto.petab.pysb_importer import PetabImporterPysb
from pypesto.sample.theano import TheanoLogProbability

from . import parameter_boundaries_scales, MODEL_FEATURE_PREFIX, load_pathway


basedir = os.path.dirname(os.path.dirname(__file__))

MODEL_FILE = os.path.join(os.path.dirname(__file__),
                          'pathway_FLT3_MAPK_AKT_STAT')


def load_petab(datafile: str, pathway_name: str):
    """
    Imports data from a csv and converts it to the petab format

    :param datafile:
        path to data csv

    :param pathway_name:
        name of pathway to use for model
    """
    data_df = pd.read_csv(datafile, index_col=[0])

    # CONDITION TABLE

    condition_table = pd.DataFrame({petab.CONDITION_ID:
                                    data_df.Sample.unique()})
    condition_table[petab.CONDITION_ID] = condition_table[
        petab.CONDITION_ID].apply(lambda x: f'sample_{x}')
    condition_table.set_index(petab.CONDITION_ID, inplace=True)

    # MEASUREMENT TABLE

    measurement_table = data_df[['Sample', 'LogFoldChange', 'site']].copy()
    measurement_table.rename(columns={
        'Sample': petab.SIMULATION_CONDITION_ID,
        'LogFoldChange': petab.MEASUREMENT,
        'site': petab.OBSERVABLE_ID,
    }, inplace=True)
    measurement_table[petab.OBSERVABLE_ID] = measurement_table[
        petab.OBSERVABLE_ID
    ].apply(lambda x: observable_id_to_model_expr(x.replace('-', '_')))
    measurement_table[petab.TIME] = np.inf

    model = load_pathway(pathway_name)

    # filter for whats available in the model:
    measurement_table = measurement_table.loc[
        measurement_table[petab.OBSERVABLE_ID].apply(
            lambda x: x in [expr.name for expr in model.expressions]
        ), :
    ]

    # OBSERVABLE TABLE

    observable_ids = measurement_table[petab.OBSERVABLE_ID].unique()

    observable_table = pd.DataFrame({
        petab.OBSERVABLE_ID: observable_ids,
        petab.OBSERVABLE_NAME: observable_ids,
        petab.OBSERVABLE_FORMULA: ['0.0' for _ in observable_ids],
    })
    observable_table[petab.NOISE_DISTRIBUTION] = 'normal'
    observable_table[petab.NOISE_FORMULA] = '1.0'
    observable_table.set_index(petab.OBSERVABLE_ID, inplace=True)

    # PARAMETER TABLE

    parameter_table = pd.DataFrame([{
        petab.PARAMETER_ID: par.name,
        petab.LOWER_BOUND: -100
        if par.name.startswith(MODEL_FEATURE_PREFIX)
        else parameter_boundaries_scales[par.name.split('_')[-1]][0],
        petab.UPPER_BOUND: 100
        if par.name.startswith(MODEL_FEATURE_PREFIX)
        else parameter_boundaries_scales[par.name.split('_')[-1]][1],
        petab.PARAMETER_SCALE: 'lin'
        if par.name.startswith(MODEL_FEATURE_PREFIX)
        else parameter_boundaries_scales[par.name.split('_')[-1]][2],
        petab.NOMINAL_VALUE: par.value,
        petab.ESTIMATE: 1
    } for par in model.parameters])

    return PetabImporterPysb(PysbPetabProblem(
        measurement_df=measurement_table,
        condition_df=condition_table,
        observable_df=observable_table,
        parameter_df=parameter_table,
        pysb_model=model
    ), output_folder=os.path.join(basedir, 'amici_models',
                                  model.name + '_petab'))


def observable_id_to_model_expr(obs_id: str):
    """
    Maps site defintions from data to model observables
    """
    phospho_site_pattern = r'_[S|Y|T][0-9]+[s|y|t]$'
    return ('p' if re.search(phospho_site_pattern, obs_id) else 't') + \
           (obs_id[:-1] if re.search(phospho_site_pattern, obs_id)
            else obs_id) + '_obs'


def load_theano(datafile: str, pathway_name: str):
    """
    loads the mechanistic model as theano operator with loss as output and
    decoder output as input

    :param datafile:
        path to data csv

    :param pathway_name:
        name of pathway to use for model
    """
    petab_importer = load_petab(datafile, pathway_name)
    pypesto_problem = petab_importer.create_problem()
    return TheanoLogProbability(pypesto_problem)
