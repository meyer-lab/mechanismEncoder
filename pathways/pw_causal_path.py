import os
import pandas as pd
from pysb import Model

from mEncoder.mechanistic_model import (
    add_monomer_synth_deg, add_activation, add_observables, add_inhibitor
)

model = Model('causal_path')

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

proteins = ['RAF1', 'MAPK3',  'ARAF', 'MAPK1', 'BRAF',
            'RPS6KA1', 'STAT5A', 'STAT5B', 'JAK1', 'JAK2', 'KSR1',
            'RPS6KB1', 'PDPK1', 'AKT1', 'AKT2', 'AKT3', 'SRC']

# filter by proteins
df = df[df.apply(
    lambda row: row.Target in proteins,
    axis=1
)]

# filter network sig

df.loc[df['Source data ID'] == 'JAK2-active-by-network-sig',
       'Source data ID'] = 'JAK2-Y1007'
df.loc[df['Source data ID'] == 'JAK1-active-by-network-sig',
       'Source data ID'] = 'JAK1-Y1034'

df = df[
    df['Source data ID'].apply(lambda x: '-by-network-sig' not in x)
]

df = df[
    df['Target data ID'].apply(lambda x: x in list(df['Source data ID']))
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
        if activator.split('-')[0] in proteins
    ]
    add_activation(model, protein, '_'.join(sites),
                   'phosphorylation', activators)

add_monomer_synth_deg('FLT3')
for protein, sites in {
    'RAF1': 'S621_S497_S296',
    'ARAF': 'S257',
}.items():
    add_activation(model, protein, sites,
                   'phosphorylation', ['FLT3'])

add_inhibitor(
    model, 'trametinib', ['ARAF', 'RAF1']
)
#add_inhibitor(
#    model, 'ruxolitinib', ['JAK1', 'JAK2']
#)
add_inhibitor(
    model, 'dasatinib', ['SRC']
)
add_inhibitor(
    model, 'gilteritinib', ['FLT3']
)

add_observables(model)
