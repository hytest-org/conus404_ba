#!/bin/bash
#SBATCH -J c404-pgw-ba_zarr
##SBATCH -t 06:00:00
#SBATCH -t 00:50:00
#SBATCH -o %j-c404-pgw-ba_zarr.out
#SBATCH -p cpu
#SBATCH -A mappnat
#SBATCH --nodes=1
#SBATCH --ntasks=1
# #SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=1
#  #SBATCH --array=10-100:10

export PATH="$PATH:$HOME/miniforge3/bin"

source activate conus404

echo $SLURM_ARRAY_TASK_ID
c404-ba_hourly_workflow process-wrf --config-file conus404-ba_hourly.yml --chunk-index ${SLURM_ARRAY_TASK_ID}