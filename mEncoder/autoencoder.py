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
from pypesto.objective import Objective

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


class MechanisticAutoEncoder(dA):
    def __init__(self,
                 n_hidden: int,
                 datafile: str,
                 pathway_name: str,):
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
        self.petab_importer = load_petab(datafile, pathway_name)
        self.pypesto_problem = self.petab_importer.create_problem()

        self.n_samples = len(self.petab_importer.petab_problem.condition_df)
        self.n_visible = len(self.petab_importer.petab_problem.observable_df)
        self.n_model_inputs = int(sum(name.startswith(MODEL_FEATURE_PREFIX)
                                      for name in
                                      self.pypesto_problem.x_names) /
                                  self.n_samples)
        self.n_kin_params = self.pypesto_problem.dim - \
            self.n_model_inputs * self.n_samples

        input_data = self.petab_importer.petab_problem.measurement_df.pivot(
            index=petab.SIMULATION_CONDITION_ID,
            columns=petab.OBSERVABLE_ID,
            values=petab.MEASUREMENT
        ).values
        super().__init__(input_data=input_data, n_hidden=n_hidden,
                         n_params=self.n_model_inputs)

        # define model theano op
        self.loss = TheanoLogProbability(self.pypesto_problem)

        # these are the kinetic parameters that are shared across all samples
        self.kin_pars = T.specify_shape(T.vector('kinetic_parameters'),
                                        (self.n_kin_params,))
        self.encoder_pars = T.specify_shape(T.vector('encoder_pars'),
                                            (self.n_encoder_pars,))

        # assemble input to model theano op
        encoded_pars = self.encode_params(self.encoder_pars)
        self.model_pars = T.concatenate([
            self.kin_pars,
            T.reshape(encoded_pars, (self.n_model_inputs * self.n_samples,))],
            axis=0
        )

    def compile_loss(self):
        return theano.function(
            [self.encoder_pars, self.kin_pars],
            self.loss(self.model_pars)
        )

    def compile_loss_grad(self):
        return theano.function(
            [self.encoder_pars, self.kin_pars],
            T.concatenate(
                [theano.grad(self.loss(self.model_pars), self.encoder_pars),
                 theano.grad(self.loss(self.model_pars), self.kin_pars)],
                axis=0
            )
        )

    def compute_inflated_pars(self, encoder_pars):
        return theano.function(
            [self.encoder_pars],
            self.encode_params(self.encoder_pars)
        )(encoder_pars)

    def generate_pypesto_objective(self):
        loss = self.compile_loss()
        loss_grad = self.compile_loss_grad()

        def fun(x: np.ndarray) -> float:
            encoder_pars = x[0:self.n_encoder_pars]
            kinetic_pars = x[self.n_encoder_pars:]
            return float(loss(encoder_pars, kinetic_pars))

        def grad(x: np.ndarray) -> np.ndarray:
            encoder_pars = x[0:self.n_encoder_pars]
            kinetic_pars = x[self.n_encoder_pars:]
            return loss_grad(encoder_pars, kinetic_pars)

        return Objective(
            fun=fun, grad=grad,
        )
