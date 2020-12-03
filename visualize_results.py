import sys
import os
import pickle
import pypesto
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import theano

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.training import create_pypesto_problem
from mEncoder import plot_and_save_fig
from mEncoder.generate_data import plot_embedding

from pypesto.visualize import waterfall, optimizer_history, \
    optimizer_convergence, parameters

MODEL = sys.argv[1]
DATA = sys.argv[2]
N_HIDDEN = int(sys.argv[3])
OPTIMIZER = sys.argv[4]

N_STARTS = 5

outfile = os.path.join('results', MODEL, DATA,
                       f'{OPTIMIZER}__{N_HIDDEN}__full.pickle')
mae = MechanisticAutoEncoder(N_HIDDEN, os.path.join('data',
                                                    f'{DATA}__{MODEL}.csv'),
                             MODEL)
problem = create_pypesto_problem(mae)

with open(outfile, 'rb') as f:
    optimizer_result, par_names = pickle.load(f)

result = pypesto.Result(problem)
for opt_result, names in zip(optimizer_result, par_names):
    sorted_par_idx = [
        names.index(name)
        for name in problem.x_names
    ]
    x_sorted = [opt_result['x'][sorted_par_idx[ix]] for ix in
                range(len(problem.x_names))]
    opt_result['x'] = x_sorted
    result.optimize_result.append(opt_result)

result.optimize_result.sort()


prefix = '__'.join([MODEL, DATA, str(N_HIDDEN), OPTIMIZER])

waterfall(result, scale_y='log10', offset_y=0.0)
plot_and_save_fig(prefix + '__waterfall.pdf')

optimizer_history(result, scale_y='log10')
plot_and_save_fig(prefix + '__optimizer_trace.pdf')

parameters(result)
plot_and_save_fig(prefix + '__parameters.pdf')

optimizer_convergence(result)
plot_and_save_fig(prefix + '__optimizer_convergence.pdf')

fig_embedding, axes_embedding = plt.subplots(1, N_STARTS,
                                             figsize=(18.5, 10.5))

embedding_fun = theano.function(
    [mae.encoder_pars],
    mae.encode(mae.encoder_pars)
)
inflate_fun = theano.function(
    [mae.encoder_pars],
    mae.encode_params(mae.encoder_pars)
)

data_dicts = []
for ir, r in enumerate(result.optimize_result.list[:N_STARTS]):
    embedding = embedding_fun(r['x'][mae.n_encoder_pars:])
    middle = int(np.floor(len(embedding) / 2))
    plot_embedding(embedding, axes_embedding[ir])

    rdatas = mae.pypesto_subproblem.objective(
        np.hstack([r['x'][mae.n_encoder_pars:],
                   inflate_fun(r['x'][:mae.n_encoder_pars]).flatten()]),
        return_dict=True
    )[
        pypesto.objective.constants.RDATAS
    ]

    for idata, rdata in enumerate(rdatas):
        ym = np.asarray(
            mae.pypesto_subproblem.objective._objectives[0].edatas[
                idata].getObservedData()
        )
        y = rdata['y']

        for iy, name in enumerate(
            mae.pypesto_subproblem.objective._objectives[0].
                    amici_model.getObservableNames()
        ):
            data_dicts.append({
                'start_index': ir,
                'observable': name,
                'data': ym[iy],
                'sim': y[0, iy]
            })

plot_and_save_fig(prefix + '__embedding.pdf')

df_data = pd.DataFrame(data_dicts)
g = sns.FacetGrid(df_data, col='observable', col_wrap=5)
g.map_dataframe(sns.scatterplot, x='data', y='sim')
plot_and_save_fig(prefix + '__fit.pdf')
