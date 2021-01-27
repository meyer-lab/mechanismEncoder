from pysb import Model

from mEncoder.mechanistic_model import (
    add_monomer_synth_deg, generate_pathway, add_activation,
    add_inhibitor
)

model = Model('ERBB_MAPK_AKT_STAT')

# EGFR
for rtkf_name in ['LE', 'L2', 'L3', 'L4']:
    add_monomer_synth_deg(rtkf_name)


erbb_cascade = [
    ('EGFR',  {'Y1173': ['LE']}),
    ('HER2',  {'Y1248': ['L2']}),
    ('HER3', {'Y1289': ['L3']}),
]
generate_pathway(model, erbb_cascade)

# STAT
stat_cascade = [
    ('STAT1', {'Y701':  ['EGFR__Y1173_p']}),
    ('Stat3', {'Y705':  ['EGFR__Y1173_p']}),
    ('STAT5-alpha', {'Y694': ['EGFR__Y1173_p']}),
    ('STAT5B', {'Y699': ['EGFR__Y1173_p']}),
]
generate_pathway(model, stat_cascade)

# ERK
for ras_name in ['HRAS', 'KRAS', 'N-Ras']:
    pass
    add_monomer_synth_deg(ras_name, nsites=['N'])
    add_activation(
        model, ras_name, 'N', 'nucleotide_exchange',
        ['EGFR__Y1173_p', 'HER2__Y1248_p', 'HER3__Y1289_p']
    )

mapk_cascade = [
    ('C-Raf',  {'S338':      ['KRAS__N_gtp', 'HRAS__N_gtp', 'N-Ras__N_gtp']}),
    ('B-Raf',  {'S445':      ['KRAS__N_gtp', 'HRAS__N_gtp', 'N-Ras__N_gtp']}),
    ('A-Raf',  {'S299':      ['KRAS__N_gtp', 'HRAS__N_gtp', 'N-Ras__N_gtp']}),
    ('MEK1',   {'S217_S221': ['C-Raf__S338_p', 'B-Raf__S445_p',
                              'A-Raf__S229_p']}),
    ('MEK2',   {'S222_S226': ['C-Raf__S338_p', 'B-Raf__S445_p',
                              'A-Raf__S229_p']}),
    ('MAPK',   {'T202_Y204': ['MEK1__S217_p__S221_p',
                              'MEK2__S222_p__S226_p']}),
]
generate_pathway(model, mapk_cascade)


# AKT
akt_cascade = [
    ('PIK3CA', {'pip2':      ['KRAS__N_gtp', 'HRAS__N_gtp', 'NRAS__N_gtp',
                              'EGFR__Y1173_p', 'HER2__Y1248_p',
                              'HER3__Y1289_p']}),
    ('Akt',    {'T308':       ['PIK3CA__pip2_p'],
                'S473':       ['PIK3CA__pip2_p']}),
]
generate_pathway(model, akt_cascade)


# GSK
gsk_cascade = [
    ('GSK3-alpha-beta', {'S21_S9': ['Akt__T308_p__S473_p']}),
    ('GSK3B', {'S9': ['Akt__T308_p__S473_p']})
]
generate_pathway(model, gsk_cascade)

# AP1
tfs = [
    ('c-Jun',  {'S73':  ['MAPK__T202_p__Y204_p', 'Akt__T308_p__S473_p']}),
    ('Elk1',   {'S383': ['MAPK__T202_p__Y204_p']}),
]

add_inhibitor(model, 'selumetinib', ['MAP2K1', 'MAP2K2'])
add_inhibitor(model, 'mk2206', ['AKT1', 'AKT2', 'AKT3'])
