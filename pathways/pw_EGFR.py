from pysb import Model

from mEncoder.mechanistic_model import add_observables
from mEncoder.pathways import add_EGFR, add_inhibitors

model = Model('EGFR')

add_EGFR(model)

add_observables(model)
add_inhibitors(model)

