import argparse
from functools import partial
import gc
import logging
import pickle
import time

from hyperopt import fmin, hp, tpe, Trials
from hyperopt.pyll import scope
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch

from data.loaders.me_loader import reformat_csv
from run_extendable import run_encoder

CV_FOLDS = 30


def obj_func(params, data):
    """
    Hyperopt objective function implemented by fmin. Calculates the loss in running the 
    autoencoder with the provided parameters.

    Parameters:
        params (dict): Dictionary of autoencoder hyper-parameters
        data (pandas.DataFrame): Data for autoencoding

    Returns:
        Average autoencoder loss over cross-validation folds
    """
    width = params['width']
    depth = params['depth']
    dropout_prob = params['dropout_prob']
    reg_coef = params['reg_coef']

    fold_means = []
    kf = KFold(n_splits=CV_FOLDS)
    for train_index, test_index in kf.split(data):
        test = data.iloc[test_index, :]
        train = data.iloc[train_index, :]
        loss = run_encoder(train, test, width, 20, depth, dropout_prob=dropout_prob, 
                           reg_coef=reg_coef)
        fold_means.append(loss)

    return np.mean(fold_means)


def main(parser):
    max_evals = parser.max_evals

    # Loads previous hyperopt trials (if available)
    try:
        trials = pickle.load(open(parser.trials_pkl, 'rb'))
        max_evals += len(trials)
    except FileNotFoundError:
        trials = Trials()
    
    # Loads data and converts to wide, samples v. features format
    sample_id = 'sample'  # Name of sample ID field
    descriptor_ids = ['Protein', 'Peptide', 'site']  # Name of sample descriptors
    value_id = 'logRatio'  # Name of sample value field
    data = reformat_csv(parser.data, sample_id, descriptor_ids, value_id, drop_axis=1)

    # Hyperparameter search space
    params = {
        'width': scope.int(hp.quniform('width', 10, 5000, q=1)),
        'depth': scope.int(hp.quniform('depth', 1, 6, q=1)),
        'dropout_prob': hp.uniform('dropout_prob', 0, 1),
        'reg_coef': hp.uniform('reg_coef', 0, 1)
    }

    # Defines objective function; partial used to permit data import
    fmin_objective = partial(obj_func, data=data)
    
    # The below code runs hyperopt for the provided number of evaluations
    # We perform only one evaluation at a time; hyperopt has a bug where it
    # fails to deallocate GPU memory between evaluations
    for evals in range(len(trials) + 1, max_evals + 1):
        fmin(
            fn=fmin_objective, 
            space=params, 
            algo=tpe.suggest, 
            max_evals=evals,
            trials=trials
        )
        torch.cuda.empty_cache()
        with open('hp_aml_trials.pkl', 'wb') as handle:
            pickle.dump(trials, handle, protocol=pickle.HIGHEST_PROTOCOL)

        time.sleep(0.01)  # Gives time for garbage collection


def _read_args():
    """
    Reads command line arguments to setup hyperopt and define input data.

    Parameters:
        None

    Returns:
        argparse.ArgumentParser with command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Specify files for classification hyper-parameter optimization'
    )
    parser.add_argument(
        '-d',
        '--data',
        dest='data',
        type=str,
        help='Data pickle file path',
    )
    parser.add_argument(
        '-m',
        '--max_evals', 
        dest='max_evals',
        type=int, 
        help='Maximum hyperopt evals per classifier',
    )
    parser.add_argument(
        '-t',
        '--trials_pkl',
        dest='trials_pkl',
        required=True,
        type=str,
        help='Pickle of previous hyperopt trials',
    )

    return parser.parse_args()


if __name__ == '__main__':
    parser = _read_args()
    main(parser)
