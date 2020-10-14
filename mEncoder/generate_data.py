from . import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import amici
import os

model, solver = load_model(force_compile=False)

np.random.seed(0)

LATENT_DIM = 5
N_SAMPLES = 20

MODEL_FEATURE_PREFIX = 'INPUT_'

boundaries = {
    'kdeg': (-3, -1),  # log10       [1/[t]]
    'eq': (1, 2),  # log10         [[c]]
    'bias': (-10, 10),  # lin      [-]
    'kcat': (1, 3),  # log10      [1/([t]*[c])]
    'scale': (-3, 0),  # log10    [1/[c]]
    'offset': (0, 1),  # log10     [[c]]
}

static_pars = dict()
for par_id in model.getParameterIds():
    if par_id.startswith(MODEL_FEATURE_PREFIX):
        continue
    lb, ub = boundaries[par_id.split('_')[-1]]
    static_pars[par_id] = np.random.random() * (ub - lb) + lb

sample_pars = [par_id for par_id in model.getParameterIds()
               if par_id.startswith(MODEL_FEATURE_PREFIX)]

decoder_mat = np.random.random((LATENT_DIM, len(sample_pars)))

model.setParameterScale(amici.parameterScalingFromIntVector([
    amici.ParameterScaling.log10
    if not par_id.startswith(MODEL_FEATURE_PREFIX)
    and not par_id.endswith('_bias')
    else amici.ParameterScaling.none
    for par_id in model.getParameterIds()
]))
model.setTimepoints([np.inf])

samples = []
while len(samples) < N_SAMPLES:
    sample_par_vals = np.random.random(LATENT_DIM).dot(decoder_mat) * 10 - 10
    sample_pars = dict(zip(sample_pars, sample_par_vals))
    for par_id, val in {**static_pars, **sample_pars}.items():
        model.setParameterById(par_id, val)

    rdata = amici.runAmiciSimulation(model, solver)
    if rdata['status'] == amici.AMICI_SUCCESS:
        sample = amici.getSimulationObservablesAsDataFrame(
            model, [amici.ExpData(model)], [rdata]
        )
        sample['Sample'] = len(samples)
        for pid, val in sample_pars.items():
            sample[pid] = val
        samples.append(sample)

df = pd.concat(samples)
df[list(model.getObservableIds())].rename(columns={
    o: o.replace('_obs', '') for o in model.getObservableIds()
}).boxplot(rot=90)
plt.show()

basedir = os.path.dirname(__file__)
formatted_df = pd.melt(df[list(model.getObservableIds()) + ['Sample']],
                       id_vars=['Sample'])
formatted_df.rename(columns={
    'variable': 'site',
    'value': 'LogFoldChange',
}, inplace=True)
formatted_df['site'] = formatted_df['site'].apply(lambda x:
                                                  x.replace('_obs', ''))
formatted_df['Gene'] = formatted_df['site'].apply(lambda x:
                                                  x.split('_')[0][1:])
formatted_df['Peptide'] = 'X.XXXXX*XXXXX.X'
formatted_df['site'] = formatted_df['site'].apply(
    lambda x: x.replace('_', '-') +
    (x.split('_')[1][0].lower() if len(x.split('_')) > 1 else '')
)
formatted_df.to_csv(os.path.join(basedir, 'data', 'synthetic_data.csv'))


