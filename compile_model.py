import sys
import os

from mEncoder.autoencoder import load_petab

DATA = sys.argv[1]
MODEL = sys.argv[2]

importer = load_petab(os.path.join('data', f'{DATA}__{MODEL}.csv'),
                      'pw_' + MODEL, 1.0)
importer.create_model(force_compile=True)
