import sys
import os

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.pretraining import (
    generate_cross_sample_pretraining_problem, pretrain
)

from pypesto.store import OptimizationResultHDF5Writer
from pypesto.visualize import waterfall, parameters

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
            col for col in df.columns if not col.startswith('INPUT_')
        ]]


sample_names = list(pretrained_samples.keys())


def startpoints(**kwargs):
    n_starts = kwargs['n_starts']
    lb = kwargs['lb']
    ub = kwargs['ub']

    dim = lb.size
    xs = np.empty((n_starts, dim))

    def make_feasible(x, ix):
        return max(min(x, ub[ix]), ub[ix])

    for istart in range(n_starts):
        # use parameter values from random start for each sample
        par_combo = pd.concat([
            pretraining[
                pretraining.index == np.random.randint(len(pretraining))]
            for pretraining in pretrained_samples.values()
        ])
        par_combo.index = sample_names
        # compute means
        means = par_combo.mean()
        # means go to average params, for inputs, take value and substract mean
        for ix, xname in enumerate(
                problem.get_reduced_vector(np.asarray(problem.x_names),
                                           problem.x_free_indices)
        ):
            if xname.startswith('INPUT_'):
                match = re.match(r'INPUT_([\w_]+)_(sample_[0-9]+)', xname)
                par = match.group(1)
                sample = match.group(2)
                xs[istart, ix] = make_feasible(
                    par_combo.loc[sample, par] - means[par], ix
                )
            else:
                xs[istart, ix] = make_feasible(means[xname], ix)

    return xs


result = pretrain(problem, startpoints, 10)

# store results
rfile = os.path.join(pretraindir, output_prefix + '.hdf5')
writer = OptimizationResultHDF5Writer(rfile)
writer.write(result, overwrite=True)

parameter_df = pd.DataFrame(
    result.optimize_result.get_for_key('x'),
    columns=problem.x_names
)
parameter_df.to_csv(os.path.join(pretraindir, output_prefix + '.csv'))

waterfall(result)
plt.tight_layout()
plt.savefig(os.path.join(pretraindir, output_prefix + '_waterfall.pdf'))

parameters(result)
plt.tight_layout()
plt.savefig(os.path.join(pretraindir, output_prefix + '_parameters.pdf'))




