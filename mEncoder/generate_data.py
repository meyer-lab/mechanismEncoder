from . import (
    load_model, parameter_boundaries_scales, MODEL_FEATURE_PREFIX,
    plot_and_save_fig
)

from .encoder import AutoEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import decomposition

import amici
import os

basedir = os.path.dirname(os.path.dirname(__file__))


def generate_synthetic_data(pathway_name: str,
                            latent_dimension: int = 2,
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
    embeddings = []
    while len(samples) < n_samples:
        # generate new fake data
        encoder.data = np.random.random(encoder.data.shape)

        if len(samples) < n_samples / 2:
            encoder.data += 1
        else:
            encoder.data -= 1

        # generate parameters from fake data
        embedding = encoder.compute_embedded_pars(tt_pars)
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
            embeddings.append(embedding)

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

    fig, ax = plt.subplots(1, 1)
    plot_embedding(np.vstack(embeddings), ax)

    plot_and_save_fig(os.path.join(
        datadir, f'synthetic__{pathway_name}__embedding.pdf'
    ))

    inputs = df[[col for col in df.columns
                 if col.startswith(MODEL_FEATURE_PREFIX)]]

    fig, ax = plt.subplots(1, 1)
    plot_pca_inputs(np.log10(inputs.values), ax)

    plot_and_save_fig(os.path.join(
        datadir, f'synthetic__{pathway_name}__input_pca.pdf'
    ))

    inputs = df[[col for col in df.columns
                 if col.startswith(MODEL_FEATURE_PREFIX) or col == 'Sample']]

    inputs = pd.melt(inputs, id_vars=['Sample'])
    inputs.index = inputs['variable'] + \
        inputs['Sample'].apply(lambda x: f'_sample_{x}')
    ref = pd.concat([pd.Series(static_pars), inputs.value])
    ref.to_csv(os.path.join(
         datadir, f'synthetic__{pathway_name}__reference_inputs.csv'
    ))

    fig, axes = plt.subplots(1, 2)
    plot_pca_inputs(df[list(model.getObservableIds())].values, axes[0],
                    axes[1])
    plot_and_save_fig(os.path.join(
        datadir, f'synthetic__{pathway_name}__data_pca.pdf'
    ))

    formatted_df.to_csv(datafile)
    return datafile


def plot_embedding(embedding: np.ndarray, ax: plt.Axes):
    middle = int(np.floor(len(embedding) / 2))
    ax.plot(embedding[:middle, 0], embedding[:middle, 1], 'k*')
    ax.plot(embedding[middle:, 0], embedding[middle:, 1], 'r*')


def plot_pca_inputs(x: np.ndarray, embed_ax: plt.Axes,
                    vexpl_ax: plt.Axes = None):
    pca = decomposition.PCA(n_components=10)
    pca.fit(x)
    x_pca = pca.transform(x)

    middle = int(np.floor(len(x) / 2))
    embed_ax.plot(x_pca[:middle, 0], x_pca[:middle, 1], 'k*')
    embed_ax.plot(x_pca[middle:, 0], x_pca[middle:, 1], 'r*')

    if vexpl_ax is not None:
        vexpl_ax.plot(np.cumsum(pca.explained_variance_ratio_))
        vexpl_ax.set_xlabel('number of components')
        vexpl_ax.set_ylabel('cumulative explained variance')

