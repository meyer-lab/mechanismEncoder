"""
Per sample pretraining.
"""

import sys
import os
import pypesto

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
for sample, problem in pretraining_problems.items():
    result = pretrain(problem, pypesto.startpoint.uniform, 10)
    output_prefix = f'{prefix}__{sample}'

    store_and_plot_pretraining(result, pretraindir, output_prefix)
    pretrained_samples.append(output_prefix + '.csv')

with open(os.path.join(pretraindir,
                       f'{prefix}.txt'), 'w') as f:
    for file in pretrained_samples:
        f.write(f'{file}\n')








