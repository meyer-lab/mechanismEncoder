from pysb import Model

from mEncoder.mechanistic_model import add_observables
from mEncoder.pathways import add_EGFR, add_MAPK, add_STAT, add_inhibitors

model = Model('EGFR_MAPK_STAT')

add_EGFR(model)
add_MAPK(model)
add_STAT(model)

add_observables(model)
add_inhibitors(model)
