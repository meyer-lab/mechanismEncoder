import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import pypesto

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.pretraining import (
    generate_patient_sample_pretraining_problems, pretrain
)

from pypesto.store import OptimizationResultHDF5Writer
from pypesto.visualize import parameters, waterfall

MODEL = sys.argv[1]
DATA = sys.argv[2]
N_HIDDEN = 1

mae = MechanisticAutoEncoder(N_HIDDEN,
                             os.path.join('data', f'{DATA}__{MODEL}.csv'),
                             MODEL)

pretraining_problems = generate_patient_sample_pretraining_problems(mae)

pretrained_samples = []
pretraindir = 'pretraining'
prefix = f'{mae.pathway_name}__{mae.data_name}'
for sample, problem in pretraining_problems.items():
    result = pretrain(problem, pypesto.startpoint.uniform, 10)
    output_prefix = f'{prefix}__{sample}'

    rfile = os.path.join(pretraindir, output_prefix + '.hdf5')
    writer = OptimizationResultHDF5Writer(rfile)
    writer.write(result, overwrite=True)

    parameter_df = pd.DataFrame(
        [r for r in result.optimize_result.get_for_key('x')
         if r is not None],
        columns=problem.x_names
    )
    parameter_df.to_csv(os.path.join(pretraindir, output_prefix + '.csv'))
    pretrained_samples.append(output_prefix + '.csv')

    waterfall(result)
    plt.tight_layout()
    plt.savefig(os.path.join(pretraindir, output_prefix + '_waterfall.pdf'))

    parameters(result)
    plt.tight_layout()
    plt.savefig(os.path.join(pretraindir, output_prefix + '_parameters.pdf'))

with open(os.path.join(pretraindir,
                       f'{prefix}.txt'), 'w') as f:
    for file in pretrained_samples:
        f.write(f'{file}\n')








