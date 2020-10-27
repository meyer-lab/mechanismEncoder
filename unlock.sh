#!/usr/bin/env bash
source venv/bin/activate
snakemake --unlock --cores 1 --config num_starts=1
