#!/usr/bin/env bash
source ./venv/bin/activate
snakemake visualize_estimation -j 2 --config num_starts=10