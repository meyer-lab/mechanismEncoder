"""
Pretraining of population + individual parameters based on per sample
pretraining
"""

import sys
import os

import pandas as pd
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import fides
import petab.visualize
import amici.petab_objective
import aesara

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.pretraining import (
    generate_cross_sample_pretraining_problem, pretrain,
    store_and_plot_pretraining
)
from mEncoder import MODEL_FEATURE_PREFIX

MODEL = sys.argv[1]
DATA = sys.argv[2]
N_HIDDEN = 3

mae = MechanisticAutoEncoder(N_HIDDEN, (
    os.path.join('data', f'{DATA}__{MODEL}__measurements.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__conditions.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__observables.tsv'),
), MODEL)

problem = generate_cross_sample_pretraining_problem(mae)
pretraindir = 'pretraining'
pretrained_samples = {}

prefix = f'{mae.pathway_name}__{mae.data_name}'
output_prefix = f'{prefix}__input'
with open(os.path.join(pretraindir, f'{prefix}.txt'), 'r') as f:
    for line in f:
        # remove linebreak which is the last character of the string
        csv = line[:-1]

        # add item to the list
        sample = os.path.splitext(csv)[0].split('__')[-1]
        df = pd.read_csv(
            os.path.join(pretraindir, csv), index_col=[0]
        )
        pretrained_samples[sample] = df[[
            col for col in df.columns
            if not col.startswith(MODEL_FEATURE_PREFIX)
        ]]


sample_names = list(pretrained_samples.keys())


def startpoints(**kwargs):
    """
    Custom startpoint routine for cross sample pretraining. This function
    uses the results computed for the completely unconstrained problem
    where the model is just fitted to each individual sample. For each
    sample, a random local optimization result is picked. Then
    shared population parameters are computed as mean over all samples and
    sample specific input parameter are computed by substracting this mean
    from the local solution.
    """
    n_starts = kwargs['n_starts']
    lb = kwargs['lb']

    dim = lb.size
    xs = np.empty((n_starts, dim))

    for istart in range(n_starts):
        # use parameter values from random start for each sample
        par_combo = pd.concat([
            pretraining[
                pretraining.index == np.random.randint(len(pretraining))
            ]
            for pretraining in pretrained_samples.values()
        ])
        par_combo.index = sample_names
        means = par_combo.mean()
        par_combo -= means
        inputs = [
            '__'.join(x.split('__')[:-1]).replace(MODEL_FEATURE_PREFIX, '')
            for x in mae.petab_importer.petab_problem.parameter_df.index
            if x.startswith(MODEL_FEATURE_PREFIX)
            and x.endswith(par_combo.index[0])
        ]
        w = la.lstsq(mae.data_pca, par_combo[inputs].values)[0].flatten()
        # compute INPUT parameters as as difference to mean
        for ix, xname in enumerate(
                problem.get_reduced_vector(np.asarray(problem.x_names),
                                           problem.x_free_indices)
        ):
            if xname.startswith('inflate') and xname.endswith('weight'):
                xs[istart, ix] = w[int(xname.split('_')[1])]
            else:
                xs[istart, ix] = means[xname]

    return xs


result = pretrain(problem, startpoints, 10,
                  subspace=fides.SubSpaceDim.TWO, maxiter=int(1e3))

store_and_plot_pretraining(result, pretraindir, output_prefix)

N_STARTS = 5

# plot residuals and pca of inputs for debugging purposes
data_dicts = []
fig_pca, axes_pca = plt.subplots(1, N_STARTS, figsize=(18.5, 10.5))


importer = mae.petab_importer
model = importer.create_model()
edatas = importer.create_edatas()

x_fun = aesara.function(
    [mae.x_embedding],
    mae.embedding_model_pars
)
x = x_fun(result.optimize_result.list[0]['x'])

simulation = amici.petab_objective.simulate_petab(
    importer.petab_problem,
    model,
    problem_parameters=dict(zip(
        mae.pypesto_subproblem.x_names, x,
    )), scaled_parameters=True,
    edatas=edatas,

)
# Convert the simulation to PEtab format.
simulation_df = amici.petab_objective.rdatas_to_simulation_df(
    simulation['rdatas'],
    model=model,
    measurement_df=importer.petab_problem.measurement_df,
)
# Plot with PEtab
petab.visualize.plot_data_and_simulation(
    exp_data=importer.petab_problem.measurement_df,
    exp_conditions=importer.petab_problem.condition_df,
    sim_data=simulation_df,
    vis_spec=importer.petab_problem.visualization_df
)
plt.tight_layout()
plt.savefig(os.path.join(
    'figures', f'pretraining_{mae.data_name}.pdf'
))

