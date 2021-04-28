"""
Per sample pretraining.
"""

import sys
import os
import fides
import pypesto
import numpy as np

import amici.petab_objective

from pypesto.optimize import FidesOptimizer

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.pretraining import (
    generate_per_sample_pretraining_problems, pretrain,
    store_and_plot_pretraining
)
from mEncoder.plotting import plot_single_sample
from mEncoder import apply_objective_settings

np.random.seed(0)

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLE = sys.argv[3]

mae = MechanisticAutoEncoder(1, (
    os.path.join('data', f'{DATA}__{MODEL}__measurements.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__conditions.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__observables.tsv'),
), MODEL, [SAMPLE])

importer = generate_per_sample_pretraining_problems(mae, SAMPLE)
pretraindir = 'pretraining'
output_prefix = f'{mae.pathway_name}__{mae.data_name}__{SAMPLE}'
problem = importer.create_problem()
model = importer.create_model()
apply_objective_settings(problem)

optimizer = FidesOptimizer(
    hessian_update=fides.HybridUpdate(),
    options={
        fides.Options.FATOL: 1e-6,
        fides.Options.XTOL: 1e-8,
        fides.Options.MAXTIME: 7200,
        fides.Options.MAXITER: 1e3,
        fides.Options.SUBSPACE_DIM: fides.SubSpaceDim.TWO,
    }
)
result = pretrain(problem, pypesto.startpoint.uniform, 10,
                  optimizer, pypesto.engine.MultiThreadEngine(4))
store_and_plot_pretraining(result, pretraindir, output_prefix)

x = problem.get_reduced_vector(result.optimize_result.list[0]['x'],
                               problem.x_free_indices)
simulation = problem.objective(x, return_dict=True)
# Convert the simulation to PEtab format.
simulation_df = amici.petab_objective.rdatas_to_simulation_df(
    simulation['rdatas'],
    model=model,
    measurement_df=importer.petab_problem.measurement_df,
)
plot_single_sample(importer.petab_problem.measurement_df,
                   simulation_df,
                   'pretraining',
                   output_prefix)
