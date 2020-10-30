import sys
import os
import pickle
import pypesto
import matplotlib.pyplot as plt
import theano

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder import plot_and_save_fig

from pypesto.visualize import waterfall, optimizer_history

MODEL = sys.argv[1]
DATA = sys.argv[2]
N_HIDDEN = int(sys.argv[3])
OPTIMIZER = sys.argv[4]

N_STARTS = 5

outfile = os.path.join('results', MODEL, DATA,
                       f'{OPTIMIZER}__{N_HIDDEN}__full.pickle')
mae = MechanisticAutoEncoder(N_HIDDEN, os.path.join('data', DATA + '.csv'),
                             MODEL)
problem = mae.create_pypesto_problem()

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

print(result.optimize_result.get_for_key('fval'))

waterfall(result, scale_y='log10')
plot_and_save_fig(
    '__'.join([MODEL, DATA, N_HIDDEN, OPTIMIZER]) + '__waterfall.pdf'
)

optimizer_history(result, scale_y='lin')
plot_and_save_fig(
    '__'.join([MODEL, DATA, N_HIDDEN, OPTIMIZER]) + '__optimizer_trace.pdf'
)

fig_embedding, axes_embedding = plt.subplots(1, N_STARTS)

embedding_fun = theano.function(
    [mae.encoder_pars],
    mae.encode(mae.encoder_pars)
)

for ir, r in enumerate(result.optimize_result.list[1:N_STARTS]):
    embedding = embedding_fun(r['x'])
    axes_embedding[ir].plot(embedding[:, 0], embedding[:, 1], 'k*')

plot_and_save_fig(
    '__'.join([MODEL, DATA, N_HIDDEN, OPTIMIZER]) + '__embedding.pdf'
)
