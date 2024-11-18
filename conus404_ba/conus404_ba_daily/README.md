# conus404_ba_daily

This directory contains the workflows for creating cloud-optimized, daily-timestep ZARR datasets from 
bias-adjusted CONUS404 hourly model output.

## Required files
```txt
conus404-ba_daily.yml
```
## Configuration file options
```yml
num_chunks_per_job: 1
target_dir: /caldera/hovenweep/projects/usgs/water/impd/pnorton/zarr_c404-bc_tmp
target_pat: target_
temp_dir: /caldera/hovenweep/projects/usgs/water/impd/pnorton/scratch_c404-bc
src_zarr: /caldera/hovenweep/projects/usgs/water/impd/pnorton/conus404-bc_hourly_TEST.zarr
dst_zarr: /caldera/hovenweep/projects/usgs/water/impd/pnorton/conus404-bc_daily_TEST.zarr
base_date: '1979-10-01 00:00:00'
end_date: '1979-10-24 23:00:00'
chunk_plan:
  time: 36
  x: 350
  y: 350
queue: cpu
account: mappnat
interface: ib0
cores_per_job: 2
processes: 2
memory_per_job: 48
max_jobs: 12
walltime: '00:40:00'
```
| Option             | Description                                                             |
|--------------------|-------------------------------------------------------------------------| 
| num_chunks_per_job | The number of chunks to process per job-array job                       |
| target_dir         | Location to store the individual chunks of processed data               |
| target_pat         | filename pattern to use for target stores                               |
| temp_dir           | Location to store temporary files created during the rechunking process |
| src_zarr           | Location of the source hourly zarr dataset                              |
| dst_zarr           | Location of the final destination zarr dataset                          |
| base_date          | The units used for the source time variable                             |
| end_date           | The last date to use when creating the destination zarr store           |
| num_days           | The number of days per chunk                                            |
| chunk_plan         | The desired chunks to use for rechunking                                |
| queue              | Name of job queue to use for job submissions                            |
| account            | Account to use for jobs                                                 |
| interface          | Network interface to use (should be ib0)                                |
| cores_per_job      | How many cpu cores to request per job                                   |
| processes          | Number of process to request                                            |
| memory_per_job     | How much memory to request per core                                     |
| max_jobs           | Maximum number of job arrays to run at a time                           |
| walltime           | Amount of walltime to request per job                                   |

## Testing
### Start interactive job
```bash
srun --pty -p cpu -A mappnat --ntasks=1 --cpus-per-task=128 --exclusive -t 08:00:00 -u bash -i
```

### Create zarr store
This will create an empty zarr dataset based on the given chunk index. The starting date for the zarr dataset is pulled from this chunk so typically you would always use the first chunk of model output for this process.
```bash
c404-ba_daily_workflow create-zarr --config-file conus404-ba_daily.yml --chunk-index=0
```
### Process source data in chunks to zarr store
The `num_chunks_per_job` variable in the configuration file controls how many chunks of model output are processed during the execution of the workflow. If this variable were set to 2 then the `chunk_index` argument would be in steps of 2 (e.g. 0, 2, 4, ...).
```bash
c404-ba_daily_workflow process-wrf --config-file conus404-ba_daily.yml --chunk-index=0
```
### Extend the time dimension in an existing zarr dataset
For the hourly workflow the `end_date` variable in the YAML file is used to indicate the date to extend the dataset to. This process is quite fast because it only updates the metadata for the zarr dataset. After the time dimension is extended then the source data chunks can be processed to fill in the values for the new time entries.
```bash
c404-ba_daily_workflow extend-time --config-file conus404-ba_daily.yml
```
