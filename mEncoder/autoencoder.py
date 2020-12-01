import os

import theano
import theano.tensor as tt

import petab

from pypesto.sample.theano import TheanoLogProbability


from . import MODEL_FEATURE_PREFIX
from .encoder import AutoEncoder
from .petab_subproblem import load_petab

TheanoFunction = theano.compile.function_module.Function

MODEL_FILE = os.path.join(os.path.dirname(__file__),
                          'pathway_FLT3_MAPK_AKT_STAT')


class MechanisticAutoEncoder(AutoEncoder):
    def __init__(self,
                 n_hidden: int,
                 datafile: str,
                 pathway_name: str,
                 par_modulation_scale: float = 1 / 2):
        """
        loads the mechanistic model as theano operator with loss as output and
        decoder output as input

        :param datafile:
            path to data csv

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
        self.data_name = os.path.splitext(os.path.basename(datafile))[0]
        self.pathway_name = pathway_name

        self.par_modulation_scale = par_modulation_scale
        self.petab_importer = load_petab(datafile, 'pw_' + pathway_name,
                                         par_modulation_scale)
        self.pypesto_subproblem = self.petab_importer.create_problem()

        self.n_samples = len(self.petab_importer.petab_problem.condition_df)
        self.n_visible = len(self.petab_importer.petab_problem.observable_df)
        self.n_model_inputs = int(sum(name.startswith(MODEL_FEATURE_PREFIX)
                                      for name in
                                      self.pypesto_subproblem.x_names) /
                                  self.n_samples)
        self.n_kin_params = \
            self.pypesto_subproblem.dim - self.n_model_inputs * self.n_samples

        input_data = self.petab_importer.petab_problem.measurement_df.pivot(
            index=petab.SIMULATION_CONDITION_ID,
            columns=petab.OBSERVABLE_ID,
            values=petab.MEASUREMENT
        )
        # zero center input data, this is equivalent to estimating biases
        # for linear autoencoders
        # https://link.springer.com/article/10.1007/BF00332918
        # https://arxiv.org/pdf/1901.08168.pdf
        input_data -= input_data.mean()

        self.sample_names = list(input_data.index)
        super().__init__(input_data=input_data.values, n_hidden=n_hidden,
                         n_params=self.n_model_inputs)

        # set tolerances
        self.pypesto_subproblem.objective._objectives[0].amici_solver\
            .setAbsoluteTolerance(1e-12)
        self.pypesto_subproblem.objective._objectives[0].amici_solver\
            .setRelativeTolerance(1e-10)
        self.pypesto_subproblem.objective._objectives[0].amici_solver\
            .setAbsoluteToleranceSteadyState(1e-10)
        self.pypesto_subproblem.objective._objectives[0].amici_solver\
            .setRelativeToleranceSteadyState(1e-8)

        # define model theano op
        self.loss = TheanoLogProbability(self.pypesto_subproblem)

        # these are the kinetic parameters that are shared across all samples
        self.kin_pars = tt.specify_shape(tt.vector('kinetic_parameters'),
                                         (self.n_kin_params,))

        self.x_names = self.x_names + [
            name for ix, name in enumerate(self.pypesto_subproblem.x_names)
            if not name.startswith(MODEL_FEATURE_PREFIX)
            and ix in self.pypesto_subproblem.x_free_indices
        ]

        # assemble input to model theano op
        encoded_pars = self.encode_params(self.encoder_pars)
        self.model_pars = tt.concatenate([
            self.kin_pars,
            tt.reshape(encoded_pars,
                       (self.n_model_inputs * self.n_samples,))],
            axis=0
        )

    def compile_loss(self) -> TheanoFunction:
        """
        Compile a theano function that evaluates the loss function
        """
        return theano.function(
            [self.encoder_pars, self.kin_pars],
            self.loss(self.model_pars)
        )

    def compile_loss_grad(self) -> TheanoFunction:
        """
        Compile a theano function that evaluates the gradient of the loss
        function
        """
        return theano.function(
            [self.encoder_pars, self.kin_pars],
            tt.concatenate(
                [theano.grad(self.loss(self.model_pars), self.encoder_pars),
                 theano.grad(self.loss(self.model_pars), self.kin_pars)],
                axis=0
            )
        )
