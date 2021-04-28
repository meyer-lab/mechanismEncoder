from pysb import Model

from mEncoder.mechanistic_model import add_observables
from mEncoder.pathways import add_EGFR, add_MAPK, add_MTOR_AKT, add_inhibitors

model = Model('EGFR_MAPK_AKT_STAT')

add_EGFR(model)
add_MAPK(model)
add_MTOR_AKT(model)

add_observables(model)
add_inhibitors(model)
