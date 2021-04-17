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
import pypesto
import petab.visualize
import amici.petab_objective

from pypesto.store import OptimizationResultHDF5Reader
from pypesto.optimize import FidesOptimizer

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.pretraining import (
    generate_cross_sample_pretraining_problem, pretrain,
    store_and_plot_pretraining
)
from mEncoder import MODEL_FEATURE_PREFIX, apply_objective_settings

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3].split('.')
INIT = sys.argv[4]
N_HIDDEN = int(sys.argv[5])
JOB = int(sys.argv[5])

np.random.seed(JOB)

mae = MechanisticAutoEncoder(N_HIDDEN, (
    os.path.join('data', f'{DATA}__{MODEL}__measurements.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__conditions.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__observables.tsv'),
), MODEL, SAMPLES, par_modulation_scale=0.5)

problem = generate_cross_sample_pretraining_problem(mae)
pretraindir = 'pretraining'
pretrained_samples = {}

prefix = f'{mae.pathway_name}__{mae.data_name}'
output_prefix = f'{prefix}__pca__{N_HIDDEN}__JOB'

if INIT == 'pca':
    for sample in SAMPLES:
        df = pd.read_csv(
            os.path.join(pretraindir, f'{prefix}__{sample}.csv'), index_col=[0]
        )
        pretrained_samples[sample] = df[[
            col for col in df.columns
            if not col.startswith(MODEL_FEATURE_PREFIX)
        ]]

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
            par_combo.index = SAMPLES
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
                    xs[istart, ix] = means[xname.replace('_obs', '')]

        return xs


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
result = pretrain(problem, startpoints, 1, optimizer)
store_and_plot_pretraining(result, pretraindir, output_prefix)

importer = mae.petab_importer
model = importer.create_model()
solver = importer.create_solver()
edatas = importer.create_edatas()

x = problem.get_reduced_vector(result.optimize_result.list[0]['x'],
                               problem.x_free_indices)
simulation = problem.objective(x, return_dict=True)

# Convert the simulation to PEtab format.
simulation_df = amici.petab_objective.rdatas_to_simulation_df(
    simulation['rdatas'],
    model=model,
    measurement_df=importer.petab_problem.measurement_df,
)

visualization_table = []

for sample in importer.petab_problem.measurement_df[
    petab.PREEQUILIBRATION_CONDITION_ID
].unique():
    measurements_condition = importer.petab_problem.measurement_df[
        importer.petab_problem.measurement_df[
            petab.PREEQUILIBRATION_CONDITION_ID
        ] == sample
    ]

    static_measurements = [
        obs_id for obs_id
        in measurements_condition[petab.OBSERVABLE_ID].unique()
        if obs_id in list(importer.petab_problem.observable_df.index)
        and (measurements_condition[
            measurements_condition[petab.OBSERVABLE_ID] == obs_id
        ][petab.TIME] == 0.0).all()
    ]
    dynamic_measurements = [
        obs_id for obs_id
        in measurements_condition[petab.OBSERVABLE_ID].unique()
        if obs_id not in static_measurements
        and obs_id in list(importer.petab_problem.observable_df.index)
    ]
    for static_obs in static_measurements:
        visualization_table.append({
            petab.PLOT_ID: f'{sample}_static',
            petab.PLOT_NAME: f'{sample} static',
            petab.Y_VALUES: static_obs,
            petab.PLOT_TYPE_SIMULATION: petab.BAR_PLOT,
            petab.LEGEND_ENTRY: f'{sample} {static_obs}',
            petab.DATASET_ID: f'{sample}'
        })

    for condition in measurements_condition[
        petab.SIMULATION_CONDITION_ID
    ].unique():
        for dynamic_obs in dynamic_measurements:
            if len(measurements_condition[
                (measurements_condition[petab.OBSERVABLE_ID] == dynamic_obs)
                & (measurements_condition[petab.SIMULATION_CONDITION_ID] ==
                   condition)
            ]) == 0:
                continue
            visualization_table.append({
                petab.PLOT_ID: f'{condition.split("__")[-1]}_{dynamic_obs}',
                petab.PLOT_NAME: f'{condition.split("__")[-1]}_{dynamic_obs}',
                petab.Y_VALUES: dynamic_obs,
                petab.PLOT_TYPE_SIMULATION: petab.LINE_PLOT,
                petab.LEGEND_ENTRY: sample,
                petab.DATASET_ID: f'{condition}',
            })

visualization_table = pd.DataFrame(visualization_table)

visualization_table[petab.Y_SCALE] = petab.LIN
visualization_table[petab.X_SCALE] = petab.LIN
visualization_table[petab.PLOT_TYPE_DATA] = petab.MEAN_AND_SD

measurement_table = importer.petab_problem.measurement_df
measurement_table[petab.DATASET_ID] = \
    measurement_table[petab.SIMULATION_CONDITION_ID]

simulation_df[petab.DATASET_ID] = simulation_df[petab.SIMULATION_CONDITION_ID]

# Plot with PEtab
axes = petab.visualize.plot_data_and_simulation(
    exp_data=measurement_table,
    exp_conditions=importer.petab_problem.condition_df,
    sim_data=simulation_df,
    vis_spec=visualization_table
)
[ax.get_legend().remove() for ax in axes.values()
 if ax.get_legend() is not None]
plt.tight_layout()
plt.savefig(os.path.join(
    'figures', f'pretraining_{mae.data_name}.pdf'
))

