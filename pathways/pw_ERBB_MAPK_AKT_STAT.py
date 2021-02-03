from pysb import Model, Observable

from mEncoder.mechanistic_model import (
    add_monomer_synth_deg, generate_pathway, add_activation,
    add_observables, add_inhibitor, add_gf_bolus
)

model = Model('ERBB_MAPK_AKT_STAT')

rtkfs = ['EGF', 'HGF', 'INSULIN', 'IGF1', 'FGF', 'NRG1']

# EGFR
for rtkf_name in rtkfs:
    add_monomer_synth_deg(rtkf_name)
    add_gf_bolus(model, rtkf_name, [rtkf_name])


erbb_cascade = [
    ('EGFR',  {'Y1173_Y992': ['EGF', 'NRG1']}),
    ('ERBB2', {'Y1248': ['EGF', 'NRG1']}),
    ('ERBB3', {'Y1289': ['NRG1']}),
    ('ERBB4', {'Y733':  ['EGF', 'NRG1']}),
    ('IGF1R',  {'Y1135_Y1136': ['INSULIN', 'IGF1']}),
    ('CMET',  {'Y1234_Y1235': ['HGF']}),
    ('FGFR',  {'Y653': ['FGF']})
]
generate_pathway(model, erbb_cascade)

active_rtks = ['EGFR__Y1173_p', 'ERBB2__Y1248_p', 'ERBB3__Y1289_p',
               'ERBB4__Y733_p', 'FGFR__Y653_p', 'IGF1R__Y1135_p__Y1136_p',
               'CMET__Y1234_p__Y1235_p']
stat_rtks = ['EGFR__Y1173_p', 'ERBB2__Y1248_p', 'ERBB3__Y1289_p',
             'ERBB4__Y733_p', 'IGF1R__Y1135_p__Y1136_p', 'FGFR__Y653_p']

# MAPK
for ras_name in ['HRAS', 'KRAS', 'NRAS']:
    add_monomer_synth_deg(ras_name, nsites=['N'])
    add_activation(
        model, ras_name, 'N', 'nucleotide_exchange',
        active_rtks
    )

mapk_cascade = [
    ('RAF1',  {'S338':      ['KRAS__N_gtp', 'HRAS__N_gtp', 'NRAS__N_gtp']}),
    ('BRAF',  {'S445':      ['KRAS__N_gtp', 'HRAS__N_gtp', 'NRAS__N_gtp']}),
    ('ARAF',  {'S299':      ['KRAS__N_gtp', 'HRAS__N_gtp', 'NRAS__N_gtp']}),
    ('MAP2K1',   {'S218_S222': ['RAF1__S338_p', 'BRAF__S445_p',
                                'ARAF__S299_p']}),
    ('MAP2K2',   {'S222_S226': ['RAF1__S338_p', 'BRAF__S445_p',
                                'ARAF__S299_p']}),
    ('MAPK1',  {'T185_Y187': ['MAP2K1__S218_p__S222_p',
                              'MAP2K2__S222_p__S226_p']}),
    ('MAPK3',  {'T202_Y204': ['MAP2K1__S218_p__S222_p',
                              'MAP2K2__S222_p__S226_p']}),
]
generate_pathway(model, mapk_cascade)
active_erk = ['MAPK1__T185_p__Y187_p', 'MAPK3__T202_p__Y204_p']
# MTOR

add_monomer_synth_deg('MTOR', asites=['C'],
                      asite_states=['c1', 'c2'])

# AKT
akt_cascade = [
    ('PIK3CA', {'pip2':      ['KRAS__N_gtp', 'HRAS__N_gtp', 'NRAS__N_gtp'] +
                              active_rtks}),
    ('AKT1',   {'T308':      ['PIK3CA__pip2_p'],
                'S473':      ['MTOR__C_c2']}),
    ('AKT2',   {'T309':      ['PIK3CA__pip2_p'],
                'S473':      ['MTOR__C_c2']}),
    ('AKT3',   {'T305':      ['PIK3CA__pip2_p'],
                'S473':      ['MTOR__C_c2']}),
]
generate_pathway(model, akt_cascade)
active_akt = ['AKT1__T308_p__S473_p', 'AKT2__T309_p__S473_p',
              'AKT3__T305_p__S473_p']

add_activation(
    model, 'MTOR', 'C', 'activation',
    active_akt,
    [],
    site_states=['c1', 'c2']
)


# GSK
gsk_cascade = [
    ('GSK3A', {'S21': active_akt}),
    ('GSK3B', {'S9': active_akt})
]
generate_pathway(model, gsk_cascade)

# STAT
stat_cascade = [
    ('STAT1', {'Y701':  stat_rtks}),
    ('STAT3', {'Y705':  stat_rtks,
               'Y727': ['GSK3B__S9_p']}),
    ('STAT5A', {'Y694': stat_rtks}),
    ('STAT5B', {'Y699': stat_rtks}),
]
generate_pathway(model, stat_cascade)


mtor_cascade = [
    ('EIF4EBP1', {'T37_T46_S65_T70':    ['MTOR__C_c1'],
                  'T37_T46':            ['GSK3B__S9_p']}),
]
generate_pathway(model, mtor_cascade)

# AP1
tfs = [
    ('JUN',  {'S73':  active_akt + active_erk}),
    ('ELK1',   {'S383': active_erk}),
]
generate_pathway(model, tfs)

# S6
s6_cascade = [
    ('RPS6', {'S245_S236': ['MTOR__C_c1'],
              'S240_S244': active_erk})  # via p70S6K
]
generate_pathway(model, s6_cascade)


Observable('AKT',
           model.monomers['AKT1']() + model.monomers['AKT2']() +
           model.monomers['AKT3']())

Observable('AKT_S473',
           model.monomers['AKT1'](S473='p') +
           model.monomers['AKT2'](S473='p') +
           model.monomers['AKT3'](S473='p'))

Observable('AKT_T308',
           model.monomers['AKT1'](T308='p') +
           model.monomers['AKT2'](T309='p') +
           model.monomers['AKT3'](T305='p'))

Observable('pMAP2K1_S217_S221',
           model.monomers['MAP2K1'](S218='p', S222='p'))

Observable('MAPK_T202',
           model.monomers['MAPK3'](T202='p'))

Observable('MAPK_T202_Y204',
           model.monomers['MAPK3'](T202='p', Y204='p'))

Observable('pGSK3_S21',
           model.monomers['GSK3A'](S21='p'))

Observable('pGSK3_S9',
           model.monomers['GSK3B'](S9='p'))

Observable('GSK3_ALPHA_BETA',
           model.monomers['GSK3B']() + model.monomers['GSK3A']())

Observable('GSK3_ALPHA_BETA_S21_S9',
           model.monomers['GSK3B'](S9='p') + model.monomers['GSK3A'](S21='p'))

Observable('STAT5_Y694',
           model.monomers['STAT5A'](Y694='p'))

add_inhibitor(model, 'MK2206', ['AKT1', 'AKT2', 'AKT3'])
add_inhibitor(model, 'TRAMETINIB', ['MAP2K1', 'MAP2K2'])
add_inhibitor(model, 'SELUMETINIB', ['MAP2K1', 'MAP2K2'])
add_inhibitor(model, 'DABRAFENIB', ['RAF1', 'BRAF', 'ARAF'])
add_inhibitor(model, 'GSK2118436', ['RAF1', 'BRAF', 'ARAF'])
add_inhibitor(model, 'PLX4720', ['RAF1', 'BRAF', 'ARAF'])
add_inhibitor(model, 'VEMURAFENIB', ['RAF1', 'BRAF', 'ARAF'])
add_inhibitor(model, 'RAPAMYCIN', ['MTOR__C_c1'])
add_inhibitor(model, 'AZD8055', ['MTOR'])
add_inhibitor(model, 'NERATINIB', ['ERBB2', 'EGFR'])
add_inhibitor(model, 'LAPATINIB', ['ERBB2', 'EGFR'])
add_inhibitor(model, 'VOXTALISIB', ['PIK3CA'])
add_inhibitor(model, 'GLEEVEC', ['PIK3CA'])

add_gf_bolus(model, 'FBS', rtkfs)
add_gf_bolus(model, 'SERUM', rtkfs)

add_observables(model)
