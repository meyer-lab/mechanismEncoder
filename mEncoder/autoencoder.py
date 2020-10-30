import os
import re
import theano
import theano.tensor as T

import pandas as pd
import numpy as np

import petab
import amici
from amici.petab_import import PysbPetabProblem
from pypesto.petab.pysb_importer import PetabImporterPysb
from pypesto.sample.theano import TheanoLogProbability
from pypesto.objective import Objective
from pypesto import Problem, HistoryOptions
from pypesto.optimize import (ScipyOptimizer, IpoptOptimizer, minimize,
                              OptimizeOptions)
import pypesto

from . import parameter_boundaries_scales, MODEL_FEATURE_PREFIX, load_pathway
from .encoder import dA

TheanoFunction = theano.compile.function_module.Function

basedir = os.path.dirname(os.path.dirname(__file__))
trace_path = os.path.join(basedir, 'traces')
TRACE_FILE_FORMAT = '{pathway}__{data}__{optimizer}__{n_hidden}__{job}__' \
                    '{{id}}.csv'

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
        pysb_model=model,
    ), output_folder=os.path.join(
        basedir, 'amici_models',
        f'{model.name}_{os.path.splitext(os.path.basename(datafile))[0]}_petab'
    ))


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
        self.data_name = os.path.splitext(os.path.basename(datafile))[0]
        self.pathway_name = pathway_name

        self.petab_importer = load_petab(datafile, 'pw_' + pathway_name)
        self.pypesto_subproblem = self.petab_importer.create_problem()

        self.n_samples = len(self.petab_importer.petab_problem.condition_df)
        self.n_visible = len(self.petab_importer.petab_problem.observable_df)
        self.n_model_inputs = int(sum(name.startswith(MODEL_FEATURE_PREFIX)
                                      for name in
                                      self.pypesto_subproblem.x_names) /
                                  self.n_samples)
        self.n_kin_params = self.pypesto_subproblem.dim - \
                            self.n_model_inputs * self.n_samples

        input_data = self.petab_importer.petab_problem.measurement_df.pivot(
            index=petab.SIMULATION_CONDITION_ID,
            columns=petab.OBSERVABLE_ID,
            values=petab.MEASUREMENT
        ).values
        super().__init__(input_data=input_data, n_hidden=n_hidden,
                         n_params=self.n_model_inputs)

        # define model theano op
        self.pypesto_subproblem.objective.amici_solver.setSensitivityMethod(
            amici.SensitivityMethod.adjoint
        )
        self.loss = TheanoLogProbability(self.pypesto_subproblem)

        # these are the kinetic parameters that are shared across all samples
        self.kin_pars = T.specify_shape(T.vector('kinetic_parameters'),
                                        (self.n_kin_params,))
        self.encoder_pars = T.specify_shape(T.vector('encoder_pars'),
                                            (self.n_encoder_pars,))

        self.x_names = [
            f'ecoder_{ip}_weight' for ip in range(self.n_encoder_pars)
        ] + [
            name for name in
            self.pypesto_subproblem.x_names
            if not name.startswith(MODEL_FEATURE_PREFIX)
        ]

        # assemble input to model theano op
        encoded_pars = self.encode_params(self.encoder_pars)
        self.model_pars = T.concatenate([
            self.kin_pars,
            T.reshape(encoded_pars, (self.n_model_inputs * self.n_samples,))],
            axis=0
        )

    def compile_loss(self) -> TheanoFunction:
        return theano.function(
            [self.encoder_pars, self.kin_pars],
            self.loss(self.model_pars)
        )

    def compile_loss_grad(self) -> TheanoFunction:
        return theano.function(
            [self.encoder_pars, self.kin_pars],
            T.concatenate(
                [theano.grad(self.loss(self.model_pars), self.encoder_pars),
                 theano.grad(self.loss(self.model_pars), self.kin_pars)],
                axis=0
            )
        )

    def compute_inflated_pars(self,
                              encoder_pars: T.vector) -> TheanoFunction:
        return theano.function(
            [self.encoder_pars],
            self.encode_params(self.encoder_pars)
        )(encoder_pars)

    def generate_pypesto_objective(self) -> Objective:
        loss = self.compile_loss()
        loss_grad = self.compile_loss_grad()

        def fun(x: np.ndarray) -> float:
            encoder_pars = x[0:self.n_encoder_pars]
            kinetic_pars = x[self.n_encoder_pars:]
            return - float(loss(encoder_pars, kinetic_pars))

        def grad(x: np.ndarray) -> np.ndarray:
            encoder_pars = x[0:self.n_encoder_pars]
            kinetic_pars = x[self.n_encoder_pars:]
            return - np.asarray(loss_grad(encoder_pars, kinetic_pars))

        return Objective(
            fun=fun, grad=grad,
        )

    def create_pypesto_problem(self) -> Problem:
        return Problem(
            objective=self.generate_pypesto_objective(),
            x_names=self.x_names,
            lb=[parameter_boundaries_scales[name.split('_')[-1]][0]
                for name in self.x_names],
            ub=[parameter_boundaries_scales[name.split('_')[-1]][1]
                for name in self.x_names],
        )

    def train(self,
              optimizer: str = 'L-BFGS-B',
              ftol: float = 1e-3,
              gtol: float = 1e-3,
              maxiter: int = 100,
              n_starts: int = 1,
              seed: int = 0):
        pypesto_problem = self.create_pypesto_problem()

        if optimizer == 'ipopt':
            opt = IpoptOptimizer(
                options={
                    'maxiter': maxiter,
                    'tol': ftol,
                    'disp': 5,
                }
            )
        else:
            opt = ScipyOptimizer(
                method=optimizer,
                options={
                    'maxiter': maxiter,
                    'ftol': ftol,
                    'gtol': gtol,
                    'disp': True,
                }
            )

        os.makedirs(trace_path, exist_ok=True)

        history_options = HistoryOptions(
            trace_record=True,
            trace_record_hess=False,
            trace_record_res=False,
            trace_record_sres=False,
            trace_record_schi2=False,
            storage_file=os.path.join(
                trace_path,
                TRACE_FILE_FORMAT.format(pathway=self.pathway_name,
                                         data=self.data_name,
                                         optimizer=optimizer,
                                         n_hidden=self.n_hidden,
                                         job=seed)
            ),
            trace_save_iter=1
        )

        np.random.seed(seed)

        optimize_options = OptimizeOptions(
            startpoint_resample=True,
            allow_failed_starts=False,
        )

        return minimize(
            pypesto_problem,
            opt,
            n_starts=n_starts,
            startpoint_method=pypesto.startpoint.uniform,
            options=optimize_options,
            history_options=history_options
        )
