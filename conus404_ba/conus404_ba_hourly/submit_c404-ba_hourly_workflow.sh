#!/bin/bash -e

MAX_JOBS=10

# The STEP value should match the num_chunks_per_job variable in the configuration YAML file
STEP=1

FIRST_STEP=0
LAST_STEP=1438
#LAST_STEP=2617

sbatch --array=${FIRST_STEP}-${LAST_STEP}:${STEP}%${MAX_JOBS} to_hourly_zarr.sbatch