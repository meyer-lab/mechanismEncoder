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
import torch

from data.loaders.cv_loader import reformat_csv
from run_CV import run_encoder

CV_FOLDS = 30


def obj_func(params, data):
    width = params['width']
    depth = params['depth']
    dropout_prob = params['dropout_prob']
    reg_coef = params['reg_coef']
    fold_size = data.shape[0] // CV_FOLDS

    fold_means = []
    for fold in range(CV_FOLDS):
        test = data.iloc[fold_size * fold:fold_size * (fold + 1), :]
        train = data.drop(range(fold_size * fold, fold_size * (fold + 1)))
        loss = run_encoder(train, test, width, depth, dropout_prob=dropout_prob, 
                           reg_coef=reg_coef)
        fold_means.append(loss)

        gc.collect()
        torch.cuda.empty_cache()

    return np.mean(fold_means)


def _read_args():
    """
    Reads command line arguments--we use these to specify pickle files for
    classification

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


def main(parser):
    max_evals = parser.max_evals
    try:
        trials = pickle.load(open(parser.trials_pkl, 'rb'))
        max_evals += len(trials)
    except FileNotFoundError:
        trials = Trials()
    
    sample_id = 'sample'  # Name of sample ID field
    descriptor_ids = ['Protein', 'Peptide', 'site']  # Name of sample descriptors
    value_id = 'logRatio'  # Name of sample value field
    data = reformat_csv(parser.data, sample_id, descriptor_ids, value_id, drop_axis=1)

    params = {
        'width': scope.int(hp.quniform('width', 10, 5000, q=1)),
        'depth': scope.int(hp.quniform('depth', 1, 6, q=1)),
        'dropout_prob': hp.uniform('dropout_prob', 0, 1),
        'reg_coef': hp.uniform('reg_coef', 0, 1)
    }

    fmin_objective = partial(obj_func, data=data)
    
    for evals in range(len(trials) + 1, max_evals + 1):
        print(torch.cuda.memory_reserved(device=None))
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

        time.sleep(0.01)


if __name__ == '__main__':
    parser = _read_args()
    main(parser)
