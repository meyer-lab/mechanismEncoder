"""
Pretraining of population + individual parameters based on per sample
pretraining
"""

import sys
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import fides

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.pretraining import (
    generate_cross_sample_pretraining_problem, pretrain,
    store_and_plot_pretraining
)
from mEncoder.generate_data import plot_pca_inputs
from mEncoder import MODEL_FEATURE_PREFIX

import pypesto

MODEL = sys.argv[1]
DATA = sys.argv[2]
N_HIDDEN = 1

mae = MechanisticAutoEncoder(N_HIDDEN,
                             os.path.join('data', f'{DATA}__{MODEL}.csv'),
                             MODEL)

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
    ub = kwargs['ub']

    dim = lb.size
    xs = np.empty((n_starts, dim))

    for istart in range(n_starts):
        # use parameter values from random start for each sample
        par_combo = pd.concat([
            pretraining[
                pretraining.index == np.random.randint(len(pretraining))]
            for pretraining in pretrained_samples.values()
        ])
        par_combo.index = sample_names
        means = par_combo.mean()
        # compute INPUT parameters as as difference to mean
        for ix, xname in enumerate(
                problem.get_reduced_vector(np.asarray(problem.x_names),
                                           problem.x_free_indices)
        ):
            if xname.startswith(MODEL_FEATURE_PREFIX):
                match = re.match(fr'{MODEL_FEATURE_PREFIX}([\w_]+)_'
                                 r'(sample_[0-9]+)', xname)
                par = match.group(1)
                sample = match.group(2)
                xs[istart, ix] = par_combo.loc[sample, par] - means[par]
            else:
                xs[istart, ix] = means[xname]

    return xs


result = pretrain(problem, startpoints, 10,
                  subspace=fides.SubSpaceDim.TWO, maxiter=int(1e2))

store_and_plot_pretraining(result, pretraindir, output_prefix)

N_STARTS = 5

# plot residuals and pca of inputs for debugging purposes
data_dicts = []
fig_pca, axes_pca = plt.subplots(1, N_STARTS, figsize=(18.5, 10.5))

for ir, r in enumerate(result.optimize_result.list[:N_STARTS]):

    rdatas = problem.objective._objectives[0](r['x'], return_dict=True)[
        pypesto.objective.constants.RDATAS
    ]

    for idata, rdata in enumerate(rdatas):
        ym = np.asarray(
            problem.objective._objectives[0].edatas[idata].getObservedData()
        )
        y = rdata['y']

        for iy, name in enumerate(
            problem.objective._objectives[0].amici_model.getObservableNames()
        ):
            data_dicts.append({
                'start_index': ir,
                'observable': name,
                'data': ym[iy],
                'sim': y[0, iy]
            })

    inputs = pd.DataFrame({
        name: val
        for name, val in zip(problem.x_names, r['x'])
        if name.startswith(MODEL_FEATURE_PREFIX)
    }, index=[0]).T

    pattern = fr'{MODEL_FEATURE_PREFIX}([\w_]+)_(sample_[0-9]+)'

    # reshape pretained input vector to sample/parameter Dataframe
    inputs['par'] = inputs.index
    inputs['sample'] = inputs.par.apply(
        lambda x: re.match(pattern, x).group(2)
    )
    inputs['par'] = inputs.par.apply(
        lambda x: re.match(pattern, x).group(1)
    )
    inputs.rename(columns={0: 'value'}, inplace=True)
    input_start = next(
        ix for ix, name in enumerate(mae.pypesto_subproblem.x_names)
        if name.startswith('INPUT_')
    )
    pretrained_inputs = pd.pivot(inputs, index='par', columns='sample').reindex(
        [re.match(pattern, name).group(1)
         for name in mae.pypesto_subproblem.x_names[
                     input_start:input_start + mae.n_model_inputs]
         ]
    )
    plot_pca_inputs(pretrained_inputs.values.T, axes_pca[ir])


plt.tight_layout()
plt.savefig(os.path.join(pretraindir, output_prefix + '__pca_inputs.pdf'))

# plot fit with data
df_data = pd.DataFrame(data_dicts)
g = sns.FacetGrid(df_data, col='observable', col_wrap=5)
g.map_dataframe(sns.scatterplot, x='data', y='sim')
plt.tight_layout()
plt.savefig(os.path.join(pretraindir, output_prefix + '__fit.pdf'))

