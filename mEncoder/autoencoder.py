import os

import aesara
import aesara.tensor as aet
import numpy as np

import petab

from pypesto.sample.theano import TheanoLogProbability

from typing import Tuple
from sklearn.decomposition import PCA

from . import MODEL_FEATURE_PREFIX
from .encoder import AutoEncoder
from .petab_subproblem import load_petab, filter_observables

AFunction = aesara.compile.Function


class MechanisticAutoEncoder(AutoEncoder):
    def __init__(self,
                 n_hidden: int,
                 datafiles: Tuple[str, str, str],
                 pathway_name: str,
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
                                         par_modulation_scale)

        full_measurements = self.petab_importer.petab_problem.measurement_df
        filter_observables(self.petab_importer.petab_problem)
        petab.lint_problem(self.petab_importer.petab_problem)

        self.pypesto_subproblem = self.petab_importer.create_problem()

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

        solver = self.pypesto_subproblem.objective._objectives[0].amici_solver
        solver.setMaxSteps(int(1e5))
        solver.setAbsoluteTolerance(1e-12)
        solver.setRelativeTolerance(1e-8)
        solver.setAbsoluteToleranceSteadyState(1e-6)
        solver.setRelativeToleranceSteadyState(1e-4)

        for e in self.pypesto_subproblem.objective._objectives[0].edatas:
            e.reinitializeFixedParameterInitialStates = True

        self.pypesto_subproblem.objective._objectives[0].guess_steadystate \
            = False

        # define model theano op
        self.loss = TheanoLogProbability(self.pypesto_subproblem)

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

    def compile_loss(self) -> AFunction:
        """
        Compile a theano function that evaluates the loss function
        """
        return aesara.function(
            [self.x],
            self.loss(self.model_pars)
        )

    def compile_embedding_loss(self) -> AFunction:
        """
        Compile a theano function that evaluates the loss function
        """
        return aesara.function(
            [self.x_embedding],
            self.loss(self.embedding_model_pars)
        )

    def compile_loss_grad(self) -> AFunction:
        """
        Compile a theano function that evaluates the gradient of the loss
        function
        """
        return aesara.function(
            [self.x],
            aesara.grad(self.loss(self.model_pars),
                        [self.x]),
        )

    def compile_embedding_loss_grad(self) -> AFunction:
        """
        Compile a theano function that evaluates the gradient of the loss
        function
        """
        return aesara.function(
            [self.x_embedding],
            aesara.grad(self.loss(self.embedding_model_pars),
                        [self.x_embedding]),
        )

    def compile_loss_hess(self) -> AFunction:
        """
        Compile a theano function that evaluates the gradient of the loss
        function
        """
        return aesara.function(
            [self.x],
            aesara.gradient.hessian(self.loss(self.model_pars),
                                    [self.x_embedding])
        )

    def compile_embedding_loss_hess(self) -> AFunction:
        """
        Compile a theano function that evaluates the gradient of the loss
        function
        """
        return aesara.function(
            [self.x_embedding],
            aesara.gradient.hessian(self.loss(self.embedding_model_pars),
                                    [self.x_embedding])
        )
