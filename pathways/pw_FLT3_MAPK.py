from mEncoder.mechanistic_model import (
    create_model, add_monomer_synth_deg, generate_pathway, add_activation,
    add_abundance_observables, add_phospho_observables
)

model = create_model('FLT3_MAPK')

# FLT3
for rtkf_name in ['FL']:
    add_monomer_synth_deg(rtkf_name)


rtk_cascade = [
    ('FLT3',  {'Y843': ['FL']}),
]
generate_pathway(model, rtk_cascade)

# ERK
for ras_name in ['HRAS', 'KRAS', 'NRAS']:
    add_monomer_synth_deg(ras_name, nsites=['N'])
    add_activation(
        model, ras_name, 'N', 'nucleotide_exchange',
        ['FLT3__Y843_p']
    )

mapk_cascade = [
    ('RAF1',   {'S338':      ['KRAS__N_gtp', 'HRAS__N_gtp', 'NRAS__N_gtp']}),
    ('BRAF',   {'S447':      ['KRAS__N_gtp', 'HRAS__N_gtp', 'NRAS__N_gtp']}),
    ('MAP2K1', {'S218_S222': ['RAF1__S338_p', 'BRAF__S447_p']}),
    ('MAP2K2', {'S222_S226': ['RAF1__S338_p', 'BRAF__S447_p']}),
    ('MAPK1', {'T185_Y187': ['MAP2K1__S218_p__S222_p',
                             'MAP2K2__S222_p__S226_p']}),
    ('MAPK3', {'T202_Y204': ['MAP2K1__S218_p__S222_p',
                             'MAP2K2__S222_p__S226_p']}),
]
generate_pathway(model, mapk_cascade)

add_abundance_observables(model)
add_phospho_observables(model)
