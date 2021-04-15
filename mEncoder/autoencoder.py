import os

import aesara
import aesara.tensor as aet
import numpy as np

import petab
import pypesto
from pypesto.hierarchical.problem import PARAMETER_TYPE

from typing import Tuple, Sequence
from sklearn.decomposition import PCA

from . import MODEL_FEATURE_PREFIX, apply_solver_settings
from .encoder import AutoEncoder
from .petab_subproblem import load_petab, filter_observables

AFunction = aesara.compile.Function


class MechanisticAutoEncoder(AutoEncoder):
    def __init__(self,
                 n_hidden: int,
                 datafiles: Tuple[str, str, str],
                 pathway_name: str,
                 samples: Sequence[str],
                 par_modulation_scale: float = 1 / 2):
        """
        loads the mechanistic model as theano operator with loss as output and
        decoder output as input

        :param datafiles:
            tuple of paths to measurements, conditions and observables files

        :param pathway_name:
            name of pathway to use for model

        :param n_hidden:
            number of nodes in the hidden layer of the encoder

        :param par_modulation_scale:
            currently this parameter only influences the strength of l2
            regularization on the inflate layer (the respective gaussian
            prior has its standard deviation defined based on the value of
            this parameter). For bounded inflate functions, this parameter
            is also intended to rescale the inputs accordingly.

        """
        self.data_name = '__'.join(
            os.path.splitext(
                os.path.basename(datafiles[0])
            )[0].split('__')[:-1]
        )
        self.pathway_name = pathway_name

        self.par_modulation_scale = par_modulation_scale
        self.petab_importer = load_petab(datafiles, 'pw_' + pathway_name,
                                         par_modulation_scale, samples)

        full_measurements = self.petab_importer.petab_problem.measurement_df
        filter_observables(self.petab_importer.petab_problem)
        petab.lint_problem(self.petab_importer.petab_problem)

        self.petab_importer.petab_problem.parameter_df[PARAMETER_TYPE] = [
            'offset' if par_id.endswith('_offset')
            else 'scaling' if par_id.endswith('_scale')
            else ''
            for par_id in self.petab_importer.petab_problem.parameter_df.index
        ]

        self.pypesto_subproblem = self.petab_importer.create_problem(
            hierarchical=True
        )

        input_data = full_measurements.loc[full_measurements.apply(
            lambda x: (x[petab.SIMULATION_CONDITION_ID] ==
                       x[petab.PREEQUILIBRATION_CONDITION_ID]) &
                      (x[petab.TIME] == 0.0), axis=1
        ), :].pivot_table(
            index=petab.SIMULATION_CONDITION_ID,
            columns=petab.OBSERVABLE_ID,
            values=petab.MEASUREMENT,
            aggfunc=np.nanmean
        )

        # extract sample names, ordering of those is important since samples
        # must match when reshaping the inflated matrix
        samples = []
        for name in self.pypesto_subproblem.x_names:
            if not name.startswith(MODEL_FEATURE_PREFIX):
                continue

            sample = name.split('__')[-1]
            if sample not in samples and sample in input_data.index:
                samples.append(sample)

        input_data = input_data.loc[samples, :]

        # remove missing values
        input_data.dropna(axis='columns', how='any', inplace=True)

        self.n_visible = input_data.shape[1]
        self.n_samples = input_data.shape[0]
        self.n_model_inputs = int(sum(name.startswith(MODEL_FEATURE_PREFIX)
                                      for name in
                                      self.pypesto_subproblem.x_names) /
                                  self.n_samples)
        self.n_kin_params = \
            self.pypesto_subproblem.dim - self.n_model_inputs * self.n_samples

        # zero center input data, this is equivalent to estimating biases
        # for linear autoencoders
        # https://link.springer.com/article/10.1007/BF00332918
        # https://arxiv.org/pdf/1901.08168.pdf
        input_data -= input_data.mean()

        self.sample_names = list(input_data.index)
        super().__init__(input_data=input_data.values, n_hidden=n_hidden,
                         n_params=self.n_model_inputs)

        # generate PCA embedding for pretraining
        pca = PCA(n_components=self.n_hidden)
        self.data_pca = pca.fit_transform(self.data)

        if isinstance(self.pypesto_subproblem.objective,
                      pypesto.objective.AmiciObjective):
            amici_objective = self.pypesto_subproblem.objective
        else:
            amici_objective = self.pypesto_subproblem.objective._objectives[0]

        apply_solver_settings(amici_objective.amici_solver)

        for e in amici_objective.edatas:
            e.reinitializeFixedParameterInitialStates = True

        amici_objective.guess_steadystate = False
        amici_objective.n_threads = 6

        self.x_names = self.x_names + [
            name for ix, name in enumerate(self.pypesto_subproblem.x_names)
            if not name.startswith(MODEL_FEATURE_PREFIX)
            and ix in self.pypesto_subproblem.x_free_indices
        ]

        # assemble input to model theano op
        self.x = aet.specify_shape(
            aet.vector('x'),
            (self.n_kin_params + self.n_encoder_pars + self.n_inflate_weights,)
        )
        encoded_pars = self.encode_params(self.x[:-self.n_kin_params])
        self.model_pars = aet.concatenate([
            self.x[-self.n_kin_params:],
            aet.reshape(encoded_pars,
                        (self.n_model_inputs * self.n_samples,))],
            axis=0
        )

        # assemble embedding to model theano op for pretraining
        self.x_embedding = aet.specify_shape(
            aet.vector('x'),
            (self.n_kin_params + self.n_model_inputs * self.n_samples,)
        )
        inflated_pars = self.inflate_params_restricted(
            self.data_pca, self.x_embedding[:-self.n_kin_params]
        )
        self.embedding_model_pars = aet.concatenate([
            self.x_embedding[-self.n_kin_params:],
            aet.reshape(inflated_pars,
                        (self.n_model_inputs * self.n_samples,))],
            axis=0
        )
