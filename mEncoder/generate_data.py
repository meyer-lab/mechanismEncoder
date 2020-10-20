from . import load_model, parameter_boundaries_scales, MODEL_FEATURE_PREFIX

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import amici
import os

basedir = os.path.dirname(os.path.dirname(__file__))


def generate_synthetic_data(latent_dimension: int = 5,
                            n_samples: int = 20) -> str:
    """
    Generates sample data using the mechanistic model.

    :param latent_dimension:
        number of latent dimensions that is used to generate the parameters
        that vary across samples

    :param n_samples:
        number of samples to generate

    :return:
        path to csv where generated data was saved
    """
    model, solver = load_model(force_compile=False)

    # setup model parameter scales
    model.setParameterScale(amici.parameterScalingFromIntVector([
        amici.ParameterScaling.none
        if par_id.startswith(MODEL_FEATURE_PREFIX)
        or parameter_boundaries_scales[par_id.split('_')[-1]][2] == 'lin'
        else amici.ParameterScaling.log10
        for par_id in model.getParameterIds()
    ]))
    # run simulations to equilibrium
    model.setTimepoints([np.inf])

    # set numpy random seed to ensure reproducibility
    np.random.seed(0)

    # generate static parameters that are consistent across samples
    static_pars = dict()
    for par_id in model.getParameterIds():
        if par_id.startswith(MODEL_FEATURE_PREFIX):
            continue
        lb, ub, _ = parameter_boundaries_scales[par_id.split('_')[-1]]
        static_pars[par_id] = np.random.random() * (ub - lb) + lb

    # identify which parameters may vary across samples
    sample_pars = [par_id for par_id in model.getParameterIds()
                   if par_id.startswith(MODEL_FEATURE_PREFIX)]

    # set up linear projection from specified latent space to parameter space
    decoder_mat = np.random.random((latent_dimension, len(sample_pars)))

    samples = []
    while len(samples) < n_samples:
        # project from low dim
        sample_par_vals = \
            np.random.random(latent_dimension).dot(decoder_mat) * 10 - 10
        sample_pars = dict(zip(sample_pars, sample_par_vals))

        # set parameters in model
        for par_id, val in {**static_pars, **sample_pars}.items():
            model.setParameterById(par_id, val)

        # run simulations, only add to samples if no integration error
        rdata = amici.runAmiciSimulation(model, solver)
        if rdata['status'] == amici.AMICI_SUCCESS:
            sample = amici.getSimulationObservablesAsDataFrame(
                model, [amici.ExpData(model)], [rdata]
            )
            sample['Sample'] = len(samples)
            for pid, val in sample_pars.items():
                sample[pid] = val
            samples.append(sample)

    # create dataframe
    df = pd.concat(samples)
    df[list(model.getObservableIds())].rename(columns={
        o: o.replace('_obs', '') for o in model.getObservableIds()
    }).boxplot(rot=90)
    plt.show()

    # format according to reference example
    formatted_df = pd.melt(df[list(model.getObservableIds()) + ['Sample']],
                           id_vars=['Sample'])
    formatted_df.rename(columns={
        'variable': 'site',
        'value': 'LogFoldChange',
    }, inplace=True)
    formatted_df['site'] = formatted_df['site'].apply(
        lambda x: x.replace('_obs', '')[1:]
    )
    formatted_df['Gene'] = formatted_df['site'].apply(
        lambda x: x.split('_')[0]
    )
    formatted_df['Peptide'] = 'X.XXXXX*XXXXX.X'
    formatted_df['site'] = formatted_df['site'].apply(
        lambda x: x.replace('_', '-') +
        (x.split('_')[1][0].lower() if len(x.split('_')) > 1 else '')
    )

    # save to csv
    datadir = os.path.join(basedir, 'data')
    os.makedirs(datadir, exist_ok=True)
    datafile = os.path.join(datadir, 'synthetic_data.csv')
    formatted_df.to_csv(datafile)
    return datafile


