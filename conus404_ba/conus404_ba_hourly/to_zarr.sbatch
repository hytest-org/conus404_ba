#!/bin/bash
#SBATCH -J c404-bc_zarr
#SBATCH -t 06:00:00
#SBATCH -o %j-c404-bc_zarr.out
#SBATCH -p cpu
#SBATCH -A mappnat
#SBATCH --nodes=1
#SBATCH --ntasks=1
# #SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24GB
#  #SBATCH --array=10-100:10

# STEP should match the stride in the --array argument
export STEP=10
export PATH="$PATH:$HOME/anaconda3/bin"

source activate pangeo

base_dir=<some_path>/wbeep/conus404_work
src_zarr_dir=<some_path>/scratch/conus404/pnorton_tmp2
dst_zarr_dir=<some_path>/scratch/conus404/conus404_hourly_biascorrected.zarr

echo $SLURM_ARRAY_TASK_ID
python <some_path>/wbeep/conus404_work/conus404_biascorrected_to_zarr.py -i $SLURM_ARRAY_TASK_ID -s $STEP -b $base_dir -z $src_zarr_dir -d $dst_zarr_dir

