import sys
import os

from mEncoder.autoencoder import MechanisticAutoEncoder


PATHWAY_MODEL = sys.argv[1]
DATA = sys.argv[2]
OPTIMIZER = sys.argv[3]
N_HIDDEN = int(sys.argv[4])
JOB = int(sys.argv[5])

mae = MechanisticAutoEncoder(N_HIDDEN,
                             os.path.join('data', DATA),
                             PATHWAY_MODEL)
result = mae.train(maxiter=int(1e3),
                   _starts=1,
                   seed=JOB,
                   optimizer=OPTIMIZER)
