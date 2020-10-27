import sys

from mEncoder.generate_data import generate_synthetic_data

DATA = sys.argv[1]

if DATA == 'synthetic':
    MODEL = 'FLT3_MAPK_AKT_STAT'
    N_HIDDEN = 5
    N_SAMPLES = 20
    data = generate_synthetic_data(MODEL, N_HIDDEN, N_SAMPLES)
