import sys
import os
import pickle

from mEncoder.autoencoder import MechanisticAutoEncoder


PATHWAY_MODEL = sys.argv[1]
DATA = sys.argv[2]
OPTIMIZER = sys.argv[3]
N_HIDDEN = int(sys.argv[4])
JOB = int(sys.argv[5])

mae = MechanisticAutoEncoder(N_HIDDEN,
                             os.path.join('data', DATA + '.csv'),
                             PATHWAY_MODEL)
result = mae.train(maxiter=int(1e3),
                   n_starts=1,
                   seed=JOB,
                   optimizer=OPTIMIZER)
outfile = os.path.join('results', PATHWAY_MODEL, DATA,
                       f'{OPTIMIZER}__{N_HIDDEN}__{JOB}.pickle')
with open(outfile, 'wb') as f:
    pickle.dump(result.optimize_result, f)

