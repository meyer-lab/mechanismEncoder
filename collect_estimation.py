import sys
import os

from pypesto.store import (
    OptimizationResultHDF5Reader, OptimizationResultHDF5Writer
)
from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.training import create_pypesto_problem
from mEncoder import results_dir
import pypesto.visualize

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]
N_HIDDEN = int(sys.argv[4])

mae = MechanisticAutoEncoder(N_HIDDEN, (
    os.path.join('data', f'{DATA}__{MODEL}__measurements.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__conditions.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__observables.tsv'),
), MODEL, SAMPLES.split('.'))
problem = create_pypesto_problem(mae)

optimizer_results = []

result_path = os.path.join(results_dir, MODEL, DATA)
result_files = os.listdir(result_path)

for file in result_files:
    if not file.startswith(f'{SAMPLES}__{N_HIDDEN}__') or \
            not file.endswith('.hdf5'):
        continue
    reader = OptimizationResultHDF5Reader(os.path.join(result_path, file))
    optimizer_results.extend(reader.read().optimize_result.list)

outfile = os.path.join(result_path, f'{SAMPLES}__{N_HIDDEN}__full.hdf5')

print(sorted([
    r['fval']
    for r in optimizer_results
])[0:min(5, len(optimizer_results))])

result = pypesto.Result(
    problem=problem
)
optimize_result = pypesto.OptimizeResult()
optimize_result.list = optimizer_results
optimize_result.sort()

result.optimize_result = optimize_result

writer = OptimizationResultHDF5Writer(outfile)
writer.write(result, overwrite=True)
