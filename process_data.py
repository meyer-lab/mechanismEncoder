import sys
import os
import re
import petab
import synapseclient
import pysb
import json

import pandas as pd
import numpy as np

import urllib.parse
import urllib.request

from mEncoder.generate_data import generate_synthetic_data
from mEncoder import load_pathway

from phosphoData.phosphoAML_PTRCdata import getAllData

basedir = os.path.dirname(os.path.dirname(__file__))


def observable_id_to_model_expr(obs_id: str,
                                dataset: str,
                                model: pysb.Model) -> str:
    """
    Maps site definitions from data to model observables

    :param obs_id:
        identifier of the phosphosite in the data table

    :param dataset:
        identifier of the dataset. Used to setup parse observable information

    :param model:
        model to which the observables are mapped

    :return:
        the name of the corresponding observable in the model
    """
    obs_id = obs_id.replace('-', '_').upper()
    if dataset == 'ptrc':
        palias = {
            r'^AKT1([\w_]*)_T309': r'AKT1\1_T308'
        }
        obs_id = re.sub(r'_?([SYT][0-9]+)[SYT]', r'_\1', obs_id)
    elif dataset == 'cppa':
        palias = {
            r'^C_RAF': 'RAF1',
            r'^B_RAF': 'BRAF',
            r'^A_RAF': 'ARAF',
            r'^MEK1': 'MAP2K1',
            r'^MEK2': 'MAP2K2',
            r'^HER2': 'ERBB2',
            r'^HER3': 'ERBB3',
            r'^STAT5_ALPHA': 'STAT5A',
            r'^C_JUN': 'JUN',
            r'^N-RAS': 'NRAS',
            r'^4E_BP1': 'EIF4EBP1',
            r'^C_MET': 'CMET',
            r'^GSK_3B': 'GSK3B',
            r'^S6': 'RPS6',
        }
        obs_id = re.sub(r'_[p|P]?([SYT][0-9]+)', r'_\1', obs_id)
    elif dataset == 'cytof':
        palias = {
            r'^P\.STAT5': 'STAT5A_Y694',
            r'^P\.MEK': 'MEK_S221',
            r'^P\.S6K$': 'RPS6KB1_S412',
            r'^P\.STAT1': 'STAT1_Y727',
            r'^P\.AKT\.SER473\.': 'AKT_S473',
            r'^P\.ERK': 'ERK_T202_Y204',
            r'^P\.HER2': 'ERBB2_Y1248',
            r'^P\.GSK3B': 'GSK3B_S9',
            r'^P\.PDPK1': 'PDPK1_S241',
            r'^P\.P90RSK': 'RPS6KA1_S380',
            r'^P\.STAT3': 'STAT3_Y705',
            r'^P\.S6$': 'RPS6_S235_S236',
            r'^P\.AKT\.THR308\.': 'AKT_T308',
            r'^P\.4EBP1': 'EIF4EBP1_T37_T46',
        }
    else:
        raise ValueError('Dataset not supported!')

    for pname, prep in palias.items():
        obs_id = re.sub(pname, prep, obs_id)

    if model.observables.get(obs_id, None):
        return obs_id

    site_pattern = r'_([S|Y|T][0-9]+)'

    monomer = re.sub(site_pattern, '', obs_id)
    sites = sorted(list(re.findall(site_pattern, obs_id)))

    name = f'p{monomer}_{"_".join(sites)}' if sites else f't{obs_id}'

    if model.observables.get(name, None):
        return name

    if model.monomers.get(monomer, None):
        print(f'could not map {obs_id} to {monomer}!')

    return ''


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

    if DATA in ['aml_ptrc']:
        syn = synapseclient.Synapse()
        syn.login()
        if DATA == 'aml_ptrc':
            df = getAllData(syn)
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

        # filter unknown cell line ~68k entries (controls
        measurement_table = measurement_table[
            measurement_table[
                petab.PREEQUILIBRATION_CONDITION_ID
            ].apply(lambda x: isinstance(x, str))
        ]

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
            if model.parameters.get(f'{pert}_0') is None:
                # remove condition
                condition_table = condition_table[
                    condition_table[petab.CONDITION_ID].apply(
                        lambda x: pert not in x.split('__')
                    )
                ]
                continue
            condition_table[f'{pert}_0'] = \
                condition_table[petab.CONDITION_ID].apply(
                    lambda x: float(int(pert in x.split('__')))
                )

        observable_mode = 'ptrc'

    if DATA in ['cppa_skin', 'cppa_breast']:
        measurement_table = pd.read_csv(
            os.path.join(
                datadir, f'{DATA}__measurements.tsv'
            ), sep='\t'
        )

        condition_table = pd.read_csv(
            os.path.join(
                datadir, f'{DATA}__conditions.tsv'
            ), sep='\t'
        )
        observable_mode = 'cppa'

    if DATA == 'dream_cytof':
        syn = synapseclient.Synapse()
        syn.login()
        df_median_phospho = pd.read_csv(syn.get('syn20613384').path)
        measurement_table_phospho = pd.melt(
            df_median_phospho,
            id_vars=['cell_line', 'treatment', 'time'],
            var_name=petab.OBSERVABLE_ID,
            value_name=petab.MEASUREMENT,
        ).rename(columns={'cell_line': petab.PREEQUILIBRATION_CONDITION_ID,
                          'time': petab.TIME})

        measurement_table_phospho[petab.PREEQUILIBRATION_CONDITION_ID] = \
            measurement_table_phospho[
                petab.PREEQUILIBRATION_CONDITION_ID
            ].apply(lambda x: f'c{x}')

        measurement_table_phospho[petab.SIMULATION_CONDITION_ID] = \
            measurement_table_phospho.apply(
                lambda x: f'{x[petab.PREEQUILIBRATION_CONDITION_ID]}__'
                          f'{x.treatment}', axis=1
            )
        measurement_table_phospho.drop(columns=['treatment'], inplace=True)

        df_proteomics = pd.read_csv(syn.get('syn20690775').path, index_col=[0])
        df_proteomics[petab.OBSERVABLE_ID] = df_proteomics.index

        df_proteomics = df_proteomics[
            df_proteomics[petab.OBSERVABLE_ID].apply(lambda x: ';' not in x)
        ]

        measurement_table_proteomics = pd.melt(
            df_proteomics,
            id_vars=[petab.OBSERVABLE_ID],
            var_name=petab.PREEQUILIBRATION_CONDITION_ID,
            value_name=petab.MEASUREMENT,
        )

        UP_ID_JSON = 'up_ids.json'
        if os.path.exists(UP_ID_JSON):
            with open(UP_ID_JSON, 'r') as fp:
                up_ids = json.load(fp)
        else:
            url = 'https://www.uniprot.org/uploadlists/'

            params = {
                'from': 'ACC+ID',
                'to': 'GENENAME',
                'format': 'tab',
                'query': ' '.join(df_proteomics[petab.OBSERVABLE_ID].unique())
            }

            data = urllib.parse.urlencode(params)
            data = data.encode('utf-8')
            req = urllib.request.Request(url, data)
            with urllib.request.urlopen(req) as f:
                response = f.read()
            up_ids = dict([
                mapping.split('\t')
                for mapping in response.decode('utf-8').split('\n')
                if '\t' in mapping
            ])
            with open(UP_ID_JSON, 'w') as fp:
                json.dump(up_ids, fp)

        measurement_table_proteomics[petab.OBSERVABLE_ID] = \
            measurement_table_proteomics[petab.OBSERVABLE_ID].apply(
                lambda x: up_ids.get(x, '')
            )

        measurement_table_proteomics = measurement_table_proteomics[
            measurement_table_proteomics[petab.OBSERVABLE_ID] != ''
        ]

        measurement_table_proteomics.dropna(axis=0, subset=[petab.MEASUREMENT],
                                            inplace=True)

        measurement_table_proteomics[petab.PREEQUILIBRATION_CONDITION_ID] = \
            measurement_table_proteomics[
                petab.PREEQUILIBRATION_CONDITION_ID
            ].apply(lambda x: f'c{x.split("_")[0]}')

        measurement_table_proteomics[petab.SIMULATION_CONDITION_ID] = \
            measurement_table_proteomics[petab.PREEQUILIBRATION_CONDITION_ID]

        measurement_table_proteomics[petab.TIME] = 0.0

        measurement_table = pd.concat([measurement_table_phospho,
                                       measurement_table_proteomics])

        measurement_table = measurement_table[
            measurement_table[petab.PREEQUILIBRATION_CONDITION_ID].apply(
                lambda x: x in ['c184A1', 'c184B5', 'cAU565', 'cBT20',
                                'cBT474']
            )
         ]

        condition_table = pd.DataFrame({
            petab.CONDITION_ID:
                measurement_table[petab.SIMULATION_CONDITION_ID].unique()
        })

        # ignore "full" for now
        condition_table = condition_table[
            condition_table[petab.CONDITION_ID].apply(
                lambda x: 'full' not in x.split('__')
            )
        ]

        perturbations = np.unique([
            p
            for c in condition_table[petab.CONDITION_ID]
            if len(c.split('__')) > 1
            for p in c.split('__')[1:] if p != 'full'
        ])
        for pert in perturbations:
            if model.parameters.get(f'{pert}_0') is None:
                # remove condition
                condition_table = condition_table[
                    condition_table[petab.CONDITION_ID].apply(
                        lambda x: pert not in x.split('__')
                    )
                ]
                continue
            condition_table[f'{pert}_0'] = \
                condition_table[petab.CONDITION_ID].apply(
                    lambda x: float(int(pert in x.split('__')))
                )

        condition_table['EGF_0'] = \
            condition_table[petab.CONDITION_ID].apply(
                lambda x: float('__' in x)
            )

        observable_mode = 'cytof'

    # filter measurements for removed conditions
    condition_ids = condition_table[petab.CONDITION_ID].unique()
    measurement_table = measurement_table[
        measurement_table.apply(
            lambda x: x[petab.SIMULATION_CONDITION_ID] in condition_ids and
                      x[petab.PREEQUILIBRATION_CONDITION_ID] in condition_ids,
            axis=1
        )
    ]

    observable_ids = [
        obs_id for obs_id in
        measurement_table.loc[:, petab.OBSERVABLE_ID].unique()
        if observable_id_to_model_expr(obs_id, observable_mode, model) != ''
    ]
    observable_table = pd.DataFrame({
        petab.OBSERVABLE_NAME: observable_ids,
    })
    observable_obs = [
        observable_id_to_model_expr(obs_id, observable_mode, model)
        for obs_id in observable_ids
    ]
    observable_table[petab.OBSERVABLE_ID] = \
        [
            f'{obs}_obs'
            for obs in observable_obs
        ]
    measurement_table[petab.OBSERVABLE_ID] = \
        measurement_table[petab.OBSERVABLE_ID].apply(
            lambda x: observable_id_to_model_expr(x, observable_mode, model)
            + '_obs'
            if observable_id_to_model_expr(x, observable_mode, model) != ''
            else x
        )
    measurement_table[petab.DATASET_ID] = measurement_table[
        petab.SIMULATION_CONDITION_ID
    ]
    observable_table[petab.OBSERVABLE_FORMULA] = \
        [
            f'log({obs}_scale * ({obs} + {obs}_offset))'
            for obs in observable_obs
        ]
    observable_table[petab.NOISE_DISTRIBUTION] = 'normal'
    observable_table[petab.NOISE_FORMULA] = \
        observable_table[petab.OBSERVABLE_ID].apply(
            lambda x: '0.1' if x.startswith('p') else '1.0'
        )

    for sample in measurement_table[
        petab.PREEQUILIBRATION_CONDITION_ID
    ].unique():
        measurements_sample = measurement_table[
            measurement_table[petab.PREEQUILIBRATION_CONDITION_ID] == sample
        ]
        conditions = [
            c for c in
            measurements_sample[petab.SIMULATION_CONDITION_ID].unique()
        ]

        visualization_table = []
        for condition in conditions:
            measurements_condition = measurements_sample[
                measurements_sample[petab.SIMULATION_CONDITION_ID] == condition
            ]

            condition_name = condition.replace(
                f'{sample}__', ''
            ).replace(sample, 'baseline')

            static_measurements = [
                obs_id for obs_id
                in measurements_condition[petab.OBSERVABLE_ID].unique()
                if obs_id in observable_table[petab.OBSERVABLE_ID].unique()
                and (measurements_condition[
                    measurements_condition[petab.OBSERVABLE_ID] == obs_id
                ][petab.TIME] == 0.0).all()
            ]
            dynamic_measurements = [
                obs_id for obs_id
                in measurements_condition[petab.OBSERVABLE_ID].unique()
                if obs_id not in static_measurements
                and obs_id in observable_table[petab.OBSERVABLE_ID].unique()
            ]
            for static_obs in static_measurements:
                visualization_table.append({
                    petab.PLOT_ID: f'{condition}_static',
                    petab.PLOT_NAME: f'{condition_name} static',
                    petab.Y_VALUES: static_obs,
                    petab.PLOT_TYPE_SIMULATION: petab.BAR_PLOT,
                    petab.LEGEND_ENTRY: f'{condition_name} {static_obs}',
                    petab.DATASET_ID: condition,
                })

            for dynamic_obs in dynamic_measurements:
                visualization_table.append({
                    petab.PLOT_ID: f'{sample}_{dynamic_obs}',
                    petab.PLOT_NAME: dynamic_obs,
                    petab.Y_VALUES: dynamic_obs,
                    petab.PLOT_TYPE_SIMULATION: petab.LINE_PLOT,
                    petab.LEGEND_ENTRY: condition_name,
                    petab.DATASET_ID: condition,
                })

        visualization_table = pd.DataFrame(visualization_table)

        visualization_table[petab.Y_SCALE] = petab.LIN
        visualization_table[petab.X_SCALE] = petab.LIN
        #visualization_table[petab.Y_LABEL] = 'measurement'
        visualization_table[petab.PLOT_TYPE_DATA] = \
            petab.MEAN_AND_SD

        visualization_file = os.path.join(
            datadir, f'{DATA}__{MODEL}__{sample}__visualization.tsv'
        )
        visualization_table.to_csv(visualization_file, sep='\t')

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



