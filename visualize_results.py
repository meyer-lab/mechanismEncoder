import sys
import os
import pypesto
import amici.petab_objective
import numpy as np
import aesara
import matplotlib.pyplot as plt

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.training import create_pypesto_problem
from mEncoder import plot_and_save_fig, results_dir, basedir
from mEncoder.plotting import plot_cross_samples

from pypesto.visualize import waterfall, optimizer_convergence
from pypesto.store import OptimizationResultHDF5Reader
from pypesto.objective.constants import FVAL

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]
N_HIDDEN = int(sys.argv[4])

N_STARTS = 5

result_path = os.path.join(results_dir, MODEL, DATA)
outfile = os.path.join(result_path, f'{SAMPLES}__{N_HIDDEN}__full.hdf5')

mae = MechanisticAutoEncoder(N_HIDDEN, (
    os.path.join('data', f'{DATA}__{MODEL}__measurements.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__conditions.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__observables.tsv'),
), MODEL, SAMPLES.split('.'))
problem = create_pypesto_problem(mae)

reader = OptimizationResultHDF5Reader(outfile)
result = pypesto.Result(problem)
result.optimize_result = reader.read().optimize_result

output_prefix = '__'.join([MODEL, DATA, SAMPLES, str(N_HIDDEN)])

waterfall(result, scale_y='log10', offset_y=0.0)
plot_and_save_fig(os.path.join(basedir, 'figures',
                               output_prefix + '__waterfall.pdf'))

optimizer_convergence(result)
plot_and_save_fig(os.path.join(basedir, 'figures',
                               output_prefix + '__optimizer_convergence.pdf'))

x = problem.get_reduced_vector(result.optimize_result.list[0]['x'],
                               problem.x_free_indices)
simulation = problem.objective(x, return_dict=True)

importer = mae.petab_importer
model = importer.create_model()

# Convert the simulation to PEtab format.
if np.isfinite(simulation[FVAL]):
    simulation_df = amici.petab_objective.rdatas_to_simulation_df(
        simulation['rdatas'],
        model=model,
        measurement_df=importer.petab_problem.measurement_df,
    )

    # Plot with PEtab
    plot_cross_samples(importer.petab_problem.measurement_df,
                       simulation_df,
                       os.path.join(basedir, 'figures'),
                       output_prefix)

embedding_fun = aesara.function(
    [mae.x], mae.embedding_fun
)

embedding = embedding_fun(result.optimize_result.list[0]['x'])


fig_embedding, axes_embedding = plt.subplots(1, 1,
                                             figsize=(18.5, 10.5))

axes_embedding.plot(embedding[:, 0], embedding[:, 1], 'bo')
axes_embedding.plot(mae.data_pca[:, 0], mae.data_pca[:, 1], 'rx')

plot_and_save_fig(output_prefix + '__embedding.pdf')
