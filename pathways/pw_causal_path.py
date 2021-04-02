import os
import pandas as pd
from pysb import Model

from mEncoder.mechanistic_model import (
    add_monomer_synth_deg, add_activation, add_observables, add_inhibitor
)

model = Model('FLT3_MAPK_cp')

cp_basedir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'causalPath', 'results',
    'fdr0.2', 'causalPathResults'
)

# merge all results
df = pd.concat([
    pd.read_csv(os.path.join(cp_basedir, cp_dir, 'results.txt'), sep='\t')
    for cp_dir in os.listdir(cp_basedir)
    if os.path.isdir(os.path.join(cp_basedir, cp_dir))
])

# ignore statistics
df = df[['Source', 'Target', 'Source data ID', 'Target data ID']]

# remove duplicates
df.drop_duplicates(inplace=True)

proteins = ['RPS6', 'RAF1', 'MAPK3', 'AKT1',  'GSK3B',  'ARAF',
            'GKS3A', 'JUNB', 'MYC', 'RB1', 'SHC1', 'JAK1', 'JAK3', 'MAPK1',
            'CREB1', 'JUND', 'JUN', 'BRAF']

# filter by proteins
df = df[df.apply(
    lambda row: row.Source in proteins and row.Target in proteins,
    axis=1
)]

# filter network sig
df = df[
    df['Source data ID'].apply(lambda x: '-by-network-sig' not in x)
]

# collect proteins
for protein in proteins:
    if protein not in list(df.Source) and protein not in list(df.Target):
        continue
    sites = set(
        site
        for data_type in ['Source', 'Target']
        for data_id in
        df[df[data_type] == protein][f'{data_type} data ID'].unique()
        for site in data_id.split('-')[1:]
    )
    add_monomer_synth_deg(protein, psites=sites)

#
for target in df['Target data ID'].unique():
    protein = target.split('-')[0]
    sites = target.split('-')[1:]
    activators = [
        '__'.join([activator.split('-')[0]] + [
            f'{site}_p' for site in activator.split('-')[1:]
        ])
        for activator in df[df['Target data ID'] == target]['Source data ID']
    ]
    add_activation(model, protein, '_'.join(sites),
                   'phosphorylation', activators)

add_inhibitor(
    model, 'trametinib', ['ARAF', 'BRAF', 'RAF1']
)

add_observables(model)
