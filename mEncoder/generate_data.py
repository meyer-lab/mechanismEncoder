from . import (
    load_model, parameter_boundaries_scales, MODEL_FEATURE_PREFIX,
    plot_and_save_fig
)

from .encoder import AutoEncoder

import numpy as np
import pandas as pd

import amici
import os

basedir = os.path.dirname(os.path.dirname(__file__))


def generate_synthetic_data(pathway_name: str,
                            latent_dimension: int = 5,
                            n_samples: int = 20) -> str:
    """
    Generates sample data using the mechanistic model.

    :param pathway_name:
        name of pathway to use for model

    :param latent_dimension:
        number of latent dimensions that is used to generate the parameters
        that vary across samples

    :param n_samples:
        number of samples to generate

    :return:
        path to csv where generated data was saved
    """
    model, solver = load_model('pw_' + pathway_name, force_compile=True)

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

    encoder = AutoEncoder(np.zeros((1, model.ny)),
                          n_hidden=latent_dimension, n_params=len(sample_pars))
    tt_pars = np.random.random(encoder.n_encoder_pars)
    for ip, name in enumerate(encoder.x_names):
        lb, ub, _ = parameter_boundaries_scales[name.split('_')[-1]]
        tt_pars[ip] = tt_pars[ip] * (ub - lb) + lb

    samples = []
    while len(samples) < n_samples:
        # generate new fake data
        encoder.data = np.random.random(encoder.data.shape)

        # generate parameters from fake data
        sample_par_vals = np.power(10, encoder.compute_inflated_pars(tt_pars))
        sample_pars = dict(zip(sample_pars, sample_par_vals[0, :]))

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
    datafile = os.path.join(datadir,
                            f'synthetic__{pathway_name}.csv')
    plot_and_save_fig(os.path.join(datadir,
                                   f'synthetic__{pathway_name}.pdf'))
    formatted_df.to_csv(datafile)
    return datafile


