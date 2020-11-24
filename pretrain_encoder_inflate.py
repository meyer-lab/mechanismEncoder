import sys
import os
import re

import pandas as pd
import matplotlib.pyplot as plt


from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.pretraining import (
    generate_encoder_inflate_pretraining_problem, pretrain
)

from pypesto.store import OptimizationResultHDF5Writer
from pypesto.visualize import waterfall, parameters
import pypesto

MODEL = sys.argv[1]
DATA = sys.argv[2]
N_HIDDEN = int(sys.argv[3])

mae = MechanisticAutoEncoder(N_HIDDEN,
                             os.path.join('data', f'{DATA}__{MODEL}.csv'),
                             MODEL)


pretraindir = 'pretraining'

prefix = f'{mae.pathway_name}__{mae.data_name}'
input_prefix = f'{prefix}__input'
output_prefix = f'{prefix}__{N_HIDDEN}__decoder_inflate'
df = pd.read_csv(os.path.join(pretraindir, f'{input_prefix}.csv'))
inputs = pd.DataFrame(
    df[[col for col in df.columns if col.startswith('INPUT_')]].mean()
)
inputs['par'] = inputs.index
inputs['sample'] = inputs.par.apply(
    lambda x: re.match(r'INPUT_([\w_]+)_(sample_[0-9]+)', x).group(2)
)
inputs['par'] = inputs.par.apply(
    lambda x: re.match(r'INPUT_([\w_]+)_(sample_[0-9]+)', x).group(1)
)
inputs.rename(columns={0: 'value'}, inplace=True)
input_start = next(
    ix for ix, name in enumerate(mae.pypesto_subproblem.x_names)
    if name.startswith('INPUT_')
)
pretrained_inputs = pd.pivot(inputs, index='par', columns='sample').reindex(
    [re.match(r'INPUT_([\w_]+)_(sample_[0-9]+)', name).group(1)
     for name in mae.pypesto_subproblem.x_names[
                 input_start:input_start + mae.n_model_inputs]
     ]
)
problem = generate_encoder_inflate_pretraining_problem(mae, pretrained_inputs)


result = pretrain(problem, pypesto.startpoint.uniform, 50, fatol=1e-4,
                  unbounded=True)
# store results
rfile = os.path.join(pretraindir, output_prefix + '.hdf5')
writer = OptimizationResultHDF5Writer(rfile)
writer.write(result, overwrite=True)

parameter_df = pd.DataFrame(
    result.optimize_result.get_for_key('x'),
    columns=problem.x_names
)
parameter_df.to_csv(os.path.join(pretraindir, output_prefix + '.csv'))

waterfall(result, scale_y='lin')
plt.tight_layout()
plt.savefig(os.path.join(pretraindir, output_prefix + '_waterfall.pdf'))

parameters(result)
plt.tight_layout()
plt.savefig(os.path.join(pretraindir, output_prefix + '_parameters.pdf'))




