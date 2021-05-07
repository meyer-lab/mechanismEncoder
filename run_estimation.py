import sys
import os

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.training import train
from mEncoder import results_dir

from pypesto.store import OptimizationResultHDF5Writer

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
    ), MODEL, SAMPLES.split('.')
)
result = train(mae, SAMPLES, n_starts=1, seed=JOB)
outdir = os.path.join(results_dir, MODEL, DATA)
outfile = os.path.join(outdir, f'{SAMPLES}__{N_HIDDEN}__{JOB}.hdf5')
os.makedirs(outdir, exist_ok=True)
if os.path.exists(outfile):
    # temp bugfix for https://github.com/ICB-DCM/pyPESTO/issues/529
    os.remove(outfile)
writer = OptimizationResultHDF5Writer(outfile)
writer.write(result)
