import sys
import os
import pickle
import re

from pypesto.optimize.optimizer import read_result_from_file
from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.training import (
    trace_path, TRACE_FILE_TEMPLATE, create_pypesto_problem
)
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

parsed_results = []

par_names = []

result_path = os.path.join('results', MODEL, DATA)
result_files = os.listdir(result_path)
trace_files = os.listdir(trace_path)

for file in trace_files:
    if re.match(TRACE_FILE_TEMPLATE.format(
        pathway=MODEL, data=f'{DATA}__{MODEL}',
        n_hidden=N_HIDDEN, job=r'[0-9]*'
    ).replace('{id}', '0'), file):
        splitted = [int(s) for s in os.path.splitext(file)[0].split('__')
                    if s.isdigit()]
        run = splitted[-2]
        start = splitted[-1]

        rfile = os.path.join(result_path,
                             f'{OPTIMIZER}__{N_HIDDEN}__{run}.pickle')
        if os.path.basename(rfile) in result_files and rfile not in \
                parsed_results:
            print(f'loading full results for run {run}')
            with open(rfile, 'rb') as f:
                result = pickle.load(f)
                # thin results
                optimizer_results += [
                    r
                    for r in result.list
                    if r['x'] is not None
                ]
                for r in result.list:
                    if r['x'] is None:
                        continue
                    if len(r['x']) < len(problem.x_names):
                        r.update_to_full(problem)
                    par_names.append([
                        problem.x_names[ix]
                        if ix in problem.x_fixed_indices
                        else result.list[0].history._trace['x'].columns[
                            problem.x_free_indices.index(ix)
                        ]
                        for ix in range(problem.dim_full)
                    ])
            parsed_results.append(rfile)

        elif rfile not in result_files:
            print(f'loading partial results for run {run}, start {start}')

            history_options = pypesto.HistoryOptions(
                trace_record=True,
                trace_record_hess=False,
                trace_record_res=False,
                trace_record_sres=False,
                trace_record_schi2=False,
                storage_file=os.path.join(
                    trace_path,
                    TRACE_FILE_TEMPLATE.format(
                        pathway=MODEL, data=f'{DATA}__{MODEL}',
                        optimizer=OPTIMIZER, n_hidden=N_HIDDEN, job=run
                    ),
                ),
                trace_save_iter=1
            )
            result = read_result_from_file(problem, history_options,
                                           str(start))

            par_names.append([
                problem.x_names[ix]
                if ix in problem.x_fixed_indices
                else result.history._trace['x'].columns[
                    problem.x_free_indices.index(ix)
                ]
                for ix in range(problem.dim_full)
            ])

            optimizer_results.append(result)


outfile = os.path.join('results', MODEL, DATA,
                       f'{OPTIMIZER}__{N_HIDDEN}__full.pickle')

print(sorted([
    r['fval']
    for r in optimizer_results
])[0:min(5, len(optimizer_results))])

with open(outfile, 'wb') as f:
    pickle.dump((optimizer_results, par_names), f)

