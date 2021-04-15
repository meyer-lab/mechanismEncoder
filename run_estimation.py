import sys
import os
import pickle

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.training import train


MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]
N_HIDDEN = int(sys.argv[4])
JOB = int(sys.argv[5])

mae = MechanisticAutoEncoder(
    N_HIDDEN, (
        os.path.join('data', f'{DATA}__{MODEL}__measurements.tsv'),
        os.path.join('data', f'{DATA}__{MODEL}__conditions.tsv'),
        os.path.join('data', f'{DATA}__{MODEL}__observables.tsv'),
    ), MODEL, SAMPLES.split(';')
)
result = train(mae, maxiter=int(1e3), n_starts=1, seed=JOB, ftol=1e-6)
outfile = os.path.join('results', MODEL, DATA,
                       f'{SAMPLES}__{N_HIDDEN}__{JOB}.pickle')

with open(outfile, 'wb') as f:
    pickle.dump(result.optimize_result, f)

