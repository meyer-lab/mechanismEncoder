"""
Pretraining of encoder/inflate parameters based on cross sample pretraining.
"""

import sys
import os
import re
import theano

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.pretraining import (
    generate_encoder_inflate_pretraining_problem, pretrain,
    store_and_plot_pretraining
)
from mEncoder import MODEL_FEATURE_PREFIX

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

# read the results from cross sample pretraining
df = pd.read_csv(os.path.join(pretraindir, f'{input_prefix}.csv'))
results = []

# iterate over local solutions from cross sample pretraining, for each local
# solution perform multiple optimization runs
for i_input in range(5):
    inputs = pd.DataFrame(
        df[[
            col for col in df.columns if col.startswith('INPUT_')
        ]].iloc[i_input, :]
    )
    # extract population parameters, these will be stored alongside
    # encoder/deflate parameters
    pars = df.loc[i_input, mae.x_names[mae.n_encoder_pars:]]

    pattern = fr'{MODEL_FEATURE_PREFIX}([\w_]+)_(sample_[0-9]+)'

    # reshape pretained input vector to sample/parameter Dataframe
    inputs['par'] = inputs.index
    inputs['sample'] = inputs.par.apply(
        lambda x: re.match(pattern, x).group(2)
    )
    inputs['par'] = inputs.par.apply(
        lambda x: re.match(pattern, x).group(1)
    )
    inputs.rename(columns={0: 'value'}, inplace=True)
    input_start = next(
        ix for ix, name in enumerate(mae.pypesto_subproblem.x_names)
        if name.startswith('INPUT_')
    )
    pretrained_inputs = pd.pivot(inputs, index='par', columns='sample').reindex(
        [re.match(pattern, name).group(1)
         for name in mae.pypesto_subproblem.x_names[
                     input_start:input_start + mae.n_model_inputs]
         ]
    )
    pretrained_inputs.columns = [r[1] for r in pretrained_inputs.columns]

    problem = generate_encoder_inflate_pretraining_problem(
        mae, pretrained_inputs, pars
    )

    result = pretrain(problem, pypesto.startpoint.uniform, 10, fatol=1e-4)
    for r in result.optimize_result.list:
        r['id'] += f'_{i_input}'

    results.append(result)

# merge results
result = pypesto.Result(problem)
result.optimize_result.list = [
    r
    for result in results
    for r in result.optimize_result.list
]
result.optimize_result.sort()


store_and_plot_pretraining(result, pretraindir, output_prefix)

# compute and plot residuals
inflate = theano.function(
    [mae.encoder_pars],
    mae.encode_params(mae.encoder_pars),
)

residuals = pd.melt(pd.concat([
    pd.DataFrame(
        inflate(r['x']),
        columns=pretrained_inputs.index,
        index=mae.sample_names,
    ) - pretrained_inputs.T
    for r in result.optimize_result.list
]))
residuals.rename(columns={'par': 'input', 'value': 'residual'}, inplace=True)
g = sns.FacetGrid(residuals, col='input', col_wrap=5)
g.map_dataframe(sns.kdeplot, x='residual')
plt.tight_layout()
plt.savefig(os.path.join(pretraindir, output_prefix + '_fit.pdf'))




