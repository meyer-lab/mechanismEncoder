import sys

from mEncoder.generate_data import generate_synthetic_data

DATA = sys.argv[1]
MODEL = sys.argv[2]

if DATA == 'synthetic':
    N_HIDDEN = 2
    N_SAMPLES = 20
    data = generate_synthetic_data(MODEL, N_HIDDEN, N_SAMPLES)
