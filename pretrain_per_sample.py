"""
Per sample pretraining.
"""

import sys
import os
import pypesto
import petab.visualize

import matplotlib.pyplot as plt

import amici.petab_objective

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.pretraining import (
    generate_per_sample_pretraining_problems, pretrain,
    store_and_plot_pretraining
)

MODEL = sys.argv[1]
DATA = sys.argv[2]
N_HIDDEN = 1

mae = MechanisticAutoEncoder(N_HIDDEN, (
    os.path.join('data', f'{DATA}__{MODEL}__measurements.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__conditions.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__observables.tsv'),
), MODEL)

pretraining_problems = generate_per_sample_pretraining_problems(mae)

pretrained_samples = []
pretraindir = 'pretraining'
prefix = f'{mae.pathway_name}__{mae.data_name}'
for sample, importer in pretraining_problems.items():
    problem = importer.create_problem()
    model = importer.create_model()
    result = pretrain(problem, pypesto.startpoint.uniform, 10)
    output_prefix = f'{prefix}__{sample}'

    store_and_plot_pretraining(result, pretraindir, output_prefix)
    pretrained_samples.append(output_prefix + '.csv')

    simulation = amici.petab_objective.simulate_petab(
        importer.petab_problem,
        model,
        problem_parameters=dict(zip(
            problem.x_names,
            result.optimize_result.list[0]['x'],
        )), scaled_parameters=True
    )
    # Convert the simulation to PEtab format.
    simulation_df = amici.petab_objective.rdatas_to_simulation_df(
        simulation['rdatas'],
        model=model,
        measurement_df=importer.petab_problem.measurement_df,
    )
    # Plot with PEtab
    ordering_cols = [
        petab.PREEQUILIBRATION_CONDITION_ID, petab.SIMULATION_CONDITION_ID,
        petab.OBSERVABLE_ID, petab.TIME
    ]
    exp_data = importer.petab_problem.measurement_df.sort_values(
        by=ordering_cols
    ).reset_index().drop(columns=['index'])
    sim_data = simulation_df.sort_values(
        by=ordering_cols
    ).reset_index().drop(columns=['index'])
    petab.visualize.plot_data_and_simulation(
        exp_data=exp_data,
        exp_conditions=importer.petab_problem.condition_df,
        sim_data=sim_data,
        vis_spec=importer.petab_problem.visualization_df
    )
    plt.tight_layout()
    plt.savefig(os.path.join(
        'figures', f'pretraining_{mae.data_name}_{sample}.pdf'
    ))

with open(os.path.join(pretraindir,
                       f'{prefix}.txt'), 'w') as f:
    for file in pretrained_samples:
        f.write(f'{file}\n')








