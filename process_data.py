import sys
import os
import re
import petab

import pandas as pd
import numpy as np

from mEncoder.generate_data import generate_synthetic_data
from mEncoder import load_pathway

basedir = os.path.dirname(os.path.dirname(__file__))


def observable_id_to_model_expr(obs_id: str) -> str:
    """
    Maps site definitions from data to model observables

    :param obs_id:
        identifier of the phosphosite in the data table

    :return:
        the name of the corresponding observable in the model
    """
    obs_id = obs_id.replace('-', '_')
    obs_id = re.sub(r'_?([SYT][0-9]+)[syt]', r'_\1', obs_id)
    phospho_site_pattern = r'_[S|Y|T][0-9]+$'
    return ('p' if re.search(phospho_site_pattern, obs_id) else 't') + \
           (obs_id if re.search(phospho_site_pattern, obs_id)
            else obs_id) + '_obs'


def convert_time_to_minutes(time_str):
    if not isinstance(time_str, str):
        return float(time_str)
    if time_str.endswith('min'):
        return float(time_str[:-4])
    if time_str.endswith('hr'):
        return float(time_str[:-3])*60


MODEL = sys.argv[1]
DATA = sys.argv[2]

datadir = os.path.join(basedir, 'data')
os.makedirs(datadir, exist_ok=True)

if DATA == 'synthetic':
    N_HIDDEN = 2
    N_SAMPLES = 20
    generate_synthetic_data(MODEL, N_HIDDEN, N_SAMPLES)

else:

    model = load_pathway('pw_' + MODEL)

    if DATA == 'cptac':
        df = pd.read_csv(os.path.join('phosphoData',
                                      'combinedPhosphoData.csv'))
        # MEASUREMENT TABLE
        # this defines the model we will later train the model on
        measurement_table = df[['logRatio', 'site', 'Gene',
                                'cellLine',
                                'timePoint', 'treatment']].copy()
        measurement_table.rename(columns={
            'logRatio': petab.MEASUREMENT,
            'site': petab.OBSERVABLE_ID,
            'timePoint': petab.TIME,
            'cellLine': petab.PREEQUILIBRATION_CONDITION_ID
        }, inplace=True)

        # filter out UR-BC.1, UR-BC.2 ~100k entries
        measurement_table = \
            measurement_table[measurement_table[
                petab.PREEQUILIBRATION_CONDITION_ID
            ].apply(lambda x: x not in ['UR-BC.1', 'UR-BC.2'])]

        # filter unknown cell line ~115k entries
        measurement_table = \
            measurement_table[measurement_table[
                petab.PREEQUILIBRATION_CONDITION_ID
            ].apply(lambda x: isinstance(x, str))]

        # make identifiers petab compatible
        measurement_table[petab.PREEQUILIBRATION_CONDITION_ID] = \
            measurement_table[petab.PREEQUILIBRATION_CONDITION_ID].apply(
                lambda x: x.replace('-', '_').replace(' ', '_')
            )

        # rename MOLM
        measurement_table.loc[
            measurement_table[petab.PREEQUILIBRATION_CONDITION_ID] == 'MOLM',
            petab.PREEQUILIBRATION_CONDITION_ID
        ] = 'MOLM_13'

        # filter Late_MOLM_13 ~113k entries
        measurement_table = measurement_table.loc[
            measurement_table[petab.PREEQUILIBRATION_CONDITION_ID]
            != 'Late_MOLM_13', :
        ]

        # convert time from string to float
        measurement_table[petab.TIME] = measurement_table[petab.TIME].apply(
            lambda x: convert_time_to_minutes(x)
        )

        # proteomics
        measurement_table.loc[
            measurement_table[petab.OBSERVABLE_ID].apply(
                lambda x: not isinstance(x, str) and np.isnan(x)
            ), petab.OBSERVABLE_ID
        ] = measurement_table.loc[
            measurement_table[petab.OBSERVABLE_ID].apply(
                lambda x: not isinstance(x, str) and np.isnan(x)
            ), 'Gene'
        ]

        # filter remaining nans ~1.2k entries
        measurement_table = measurement_table.loc[
            measurement_table[petab.OBSERVABLE_ID].apply(
                lambda x: isinstance(x, str)
            ), :
        ]

        # match observables with model expressions
        observable_ids = measurement_table.loc[
            measurement_table[petab.OBSERVABLE_ID].apply(
                lambda x: observable_id_to_model_expr(x) in [
                    expr.name for expr in model.expressions
                ]
            ), petab.OBSERVABLE_ID
        ].unique()

        # OBSERVABLE TABLE
        # this defines how model simulation are linked to experimental data,
        # currently this uses quantities that were already defined in the model
        observable_table = pd.DataFrame({
            petab.OBSERVABLE_ID: observable_ids,
            petab.OBSERVABLE_NAME: observable_ids,
            petab.OBSERVABLE_FORMULA: [
                observable_id_to_model_expr(obs) for obs in observable_ids
            ],
        })
        observable_table[petab.NOISE_DISTRIBUTION] = 'normal'
        observable_table[petab.NOISE_FORMULA] = '1.0'

        measurement_table.loc[
            (measurement_table['treatment'] == 'no treatment') |
            (measurement_table['treatment'] == 'No treatment'),
            'treatment'
        ] = 'none'

        measurement_table['treatment'] = measurement_table['treatment'].apply(
            lambda x: x.replace(
                'Trametinib', 'trametinib'
            ).replace('+', '__').replace('-', '_').replace(' ', '_')
        )

        measurement_table[petab.SIMULATION_CONDITION_ID] = \
            measurement_table.apply(
                lambda row: row[petab.PREEQUILIBRATION_CONDITION_ID]
                if row.treatment == 'none' else
                '__'.join([row[petab.PREEQUILIBRATION_CONDITION_ID],
                           row.treatment]),
                axis=1
            )

        measurement_table.drop(columns=['Gene', 'treatment'],
                               inplace=True)

        condition_table = pd.DataFrame({
            petab.CONDITION_ID:
                measurement_table[petab.SIMULATION_CONDITION_ID].unique()
        })
        perturbations = np.unique([
            p
            for c in condition_table[petab.CONDITION_ID]
            if len(c.split('__')) > 1
            for p in c.split('__')[1:]
        ])
        for pert in perturbations:
            if model.components.get(pert) is None:
                continue  # perturbation not available
            condition_table[pert] = condition_table[petab.CONDITION_ID].apply(
                lambda x: float(int(pert in x.split('__')))
            )

    measurement_file = os.path.join(
            datadir, f'{DATA}__{MODEL}__measurements.tsv'
        )
    measurement_table.to_csv(measurement_file, sep='\t')

    condition_file = os.path.join(
        datadir, f'{DATA}__{MODEL}__conditions.tsv'
    )
    condition_table.set_index(petab.CONDITION_ID, inplace=True)
    condition_table.to_csv(condition_file, sep='\t')

    observable_file = os.path.join(
        datadir, f'{DATA}__{MODEL}__observables.tsv'
    )
    observable_table.set_index(petab.OBSERVABLE_ID, inplace=True)
    observable_table.to_csv(observable_file, sep='\t')
