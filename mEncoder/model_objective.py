import os
import re
import theano
import theano.tensor as T

import pandas as pd
import numpy as np

import petab
from amici.petab_import import PysbPetabProblem
from pypesto.petab.pysb_importer import PetabImporterPysb
from pypesto.sample.theano import TheanoLogProbability

from . import parameter_boundaries_scales, MODEL_FEATURE_PREFIX, load_pathway
from .encoder import dA


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

    model = load_pathway(pathway_name)

    features = [par for par in model.parameters
                if par.name.startswith(MODEL_FEATURE_PREFIX)]

    def condition_id_from_sample(cond_id):
        return f'sample_{cond_id}'

    # CONDITION TABLE
    conditions = {
        petab.CONDITION_ID:  [condition_id_from_sample(x)
                              for x in data_df.Sample.unique()]
    }
    for feature in features:
        conditions[feature.name] = [
            f'{feature.name}_{s}' for s in conditions[petab.CONDITION_ID]
        ]

    condition_table = pd.DataFrame(conditions).set_index(petab.CONDITION_ID)

    # MEASUREMENT TABLE

    measurement_table = data_df[['Sample', 'LogFoldChange', 'site']].copy()
    measurement_table.rename(columns={
        'Sample': petab.SIMULATION_CONDITION_ID,
        'LogFoldChange': petab.MEASUREMENT,
        'site': petab.OBSERVABLE_ID,
    }, inplace=True)
    measurement_table[petab.SIMULATION_CONDITION_ID] = \
        measurement_table[petab.SIMULATION_CONDITION_ID].apply(
            condition_id_from_sample
        )
    measurement_table[petab.OBSERVABLE_ID] = measurement_table[
        petab.OBSERVABLE_ID
    ].apply(lambda x: observable_id_to_model_expr(x.replace('-', '_')))
    measurement_table[petab.TIME] = np.inf

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
    }).set_index(petab.OBSERVABLE_ID)
    observable_table[petab.NOISE_DISTRIBUTION] = 'normal'
    observable_table[petab.NOISE_FORMULA] = '1.0'

    # PARAMETER TABLE
    params = [par for par in model.parameters
              if not par.name.startswith(MODEL_FEATURE_PREFIX)]
    param_defs = [{
        petab.PARAMETER_ID: par.name,
        petab.LOWER_BOUND: parameter_boundaries_scales[
            par.name.split('_')[-1]][0],
        petab.UPPER_BOUND: parameter_boundaries_scales[
            par.name.split('_')[-1]][1],
        petab.PARAMETER_SCALE: parameter_boundaries_scales[
            par.name.split('_')[-1]][2],
        petab.NOMINAL_VALUE: par.value,
        petab.ESTIMATE: 1
    } for par in params]

    for cond in condition_table.index.values:
        param_defs.extend([{
            petab.PARAMETER_ID: f'{par.name}_{cond}',
            petab.LOWER_BOUND: -100,
            petab.UPPER_BOUND: 100,
            petab.PARAMETER_SCALE: 'lin',
            petab.NOMINAL_VALUE: par.value,
            petab.ESTIMATE: 1
        } for par in features])

    parameter_table = pd.DataFrame(param_defs).set_index(petab.PARAMETER_ID)

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


def load_theano(datafile: str,
                pathway_name: str,
                n_hidden: int = 10):
    """
    loads the mechanistic model as theano operator with loss as output and
    decoder output as input

    :param datafile:
        path to data csv

    :param pathway_name:
        name of pathway to use for model

    :param n_hidden:
        number of nodes in the hidden layer of the encoder
    """
    petab_importer = load_petab(datafile, pathway_name)
    pypesto_problem = petab_importer.create_problem()

    n_samples = len(petab_importer.petab_problem.condition_df)
    n_visible = len(petab_importer.petab_problem.observable_df)
    n_model_inputs = int(sum(name.startswith(MODEL_FEATURE_PREFIX)
                             for name in pypesto_problem.x_names)/n_samples)
    n_kin_params = pypesto_problem.dim - n_model_inputs

    encoder = dA(n_visible=n_visible,
                 n_hidden=n_hidden,
                 n_params=n_model_inputs)

    # define model theano op
    loss = TheanoLogProbability(pypesto_problem)

    # encode data
    data = theano.shared(np.zeros((n_samples, n_visible),
                                  dtype=theano.config.floatX),
                         name='data_input')

    # these are the kinetic parameters that are shared across all samples
    kin_pars = T.specify_shape(T.vector('kinetic_parameters'),
                               (n_kin_params,))

    # assemble input to model theano op
    encoded_pars = encoder.encode_params(data)
    model_pars = T.concatenate([
        kin_pars, T.reshape(encoded_pars, (n_model_inputs * n_samples,))],
        axis=0
    )
    return theano.function(
        [encoder.encoder_pars, encoder.inflate_pars, kin_pars],
        loss(model_pars)
    )
