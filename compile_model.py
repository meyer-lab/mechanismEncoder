import sys

from mEncoder.autoencoder import load_petab

DATA = sys.argv[1]
MODEL = sys.argv[2]

data = load_petab(DATA, MODEL)
