import sys
import os

from mEncoder.autoencoder import load_petab

DATA = sys.argv[1]
MODEL = sys.argv[2]

importer = load_petab(os.path.join('data', DATA + '.csv'), 'pw_' + MODEL)
importer.create_model(force_compile=True)
