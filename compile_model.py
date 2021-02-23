import sys
import os

from mEncoder.autoencoder import load_petab

MODEL = sys.argv[1]
DATA = sys.argv[2]

importer = load_petab((
    os.path.join('data', f'{DATA}__{MODEL}__measurements.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__conditions.tsv'),
    os.path.join('data', f'{DATA}__{MODEL}__observables.tsv'),
), 'pw_' + MODEL, 1.0)
importer.create_model(force_compile=True)

