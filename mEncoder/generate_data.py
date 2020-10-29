from . import load_model, parameter_boundaries_scales, MODEL_FEATURE_PREFIX
from .encoder import dA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import amici
import os
import theano
import theano.tensor as T

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
    model, solver = load_model('pw_' + pathway_name, force_compile=False)

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
    encoder = dA(np.zeros((n_samples, 10)),
                 n_hidden=latent_dimension, n_params=len(sample_pars))
    tt_pars = np.random.random(encoder.n_encoder_pars)
    for ip, name in enumerate(encoder.x_names):
        lb, ub, _ = parameter_boundaries_scales[name.split('_')[-1]]
        tt_pars[ip] = tt_pars[ip] * (ub - lb) + lb

    tt_data = T.specify_shape(T.vector('embedded_data'),
                              (encoder.n_hidden,))
    inflate_fun = theano.function(
        [tt_data], encoder.inflate_params(tt_data, tt_pars)
    )

    samples = []
    while len(samples) < n_samples:
        # project from low dim
        embedded_sample_pars = np.random.random(latent_dimension) * 10 - 5
        sample_par_vals = inflate_fun(embedded_sample_pars,)
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
    datafile = os.path.join(datadir, 'synthetic.csv')
    formatted_df.to_csv(datafile)
    return datafile


