import sys
import os
import pickle
import re

import pandas as pd

from pypesto.optimize.optimizer import read_result_from_file
from mEncoder.autoencoder import MechanisticAutoEncoder, trace_path
import pypesto.visualize

MODEL = sys.argv[1]
DATA = sys.argv[2]
N_HIDDEN = int(sys.argv[3])
OPTIMIZER = sys.argv[4]

mae = MechanisticAutoEncoder(N_HIDDEN, os.path.join('data', DATA + '.csv'),
                             MODEL)
problem = mae.create_pypesto_problem()

optimizer_results = []

parsed_results = []

par_names = []

result_path = os.path.join('results', MODEL, DATA)
result_files = os.listdir(result_path)
trace_files = os.listdir(trace_path)

for file in trace_files:
    if re.match(f'{OPTIMIZER}__{N_HIDDEN}__[0-9]*__0.csv', file):
        splitted = [int(s) for s in file.split('__') if s.isdigit()]
        run = splitted[0]
        start = splitted[1]

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
                    f'{OPTIMIZER}__{N_HIDDEN}__{run}__{{id}}.csv',
                ),
                trace_save_iter=1
            )
            try:
                result = read_result_from_file(problem, history_options,
                                               str(start))
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                print(f'corrupt file '
                      f'{OPTIMIZER}__{N_HIDDEN}__{run}__{{id}}.csv',)
                continue

            par_names.append([
                problem.x_names[ix]
                if ix in problem.x_fixed_indices
                else result.history._trace['x'].columns[
                    problem.x_free_indices.index(ix)
                ]
                for ix in range(problem.dim_full)
            ])

            optimizer_results.append(result)


outfile = os.path.join('results', '{model}', '{data}',
                       f'{OPTIMIZER}__{N_HIDDEN})__full.pickle')

print(sorted([
    r['fval']
    for r in optimizer_results
])[0:min(5, len(optimizer_results))])

with open(outfile, 'wb') as f:
    pickle.dump((optimizer_results, par_names), f)

