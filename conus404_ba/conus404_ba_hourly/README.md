# conus404_ba_hourly

The original conus404-ba workflow in 2023 used:
- `conus404_rechunk_bias_corrected.py`
- `conus404_biascorrected_to_zarr.py`

An updated workflow, `c404_ba_hourly_workflow.py`, was created in 2024 that includes workflow
commands to create an empty hourly zarr dataset, rechunk the bias-corrected data, and write the
rechunked data to the hourly zarr dataset.

## Required files
```txt
conus404-ba_hourly_metadata.csv
conus404-ba_hourly.yml
var_list.csv
```
## Configuration file options
```yml
num_chunks_per_job: 1
wrf_dir: /caldera/hovenweep/projects/usgs/water/impd/hytest/conus404-tpBiasCorr
wrf_file_pat: '{wrf_dir}/{wy_dir}/{fdate.strftime("%Y%m%d%H%M")}.LDASIN_DOMAIN1'
target_dir: <path>/zarr_c404-bc_tmp
target_pat: target_
temp_dir: <path>/scratch_c404-bc
metadata_file: /caldera/hovenweep/projects/usgs/water/impd/hytest/working/src_files/conus404-ba_hourly/conus404-ba_hourly_metadata.csv
vars_file: /caldera/hovenweep/projects/usgs/water/impd/hytest/working/src_files/conus404-ba_hourly/var_list.csv
dst_zarr: <path>/conus404-bc_hourly_TEST.zarr
base_date: '1979-10-01 00:00:00'
end_date: '1979-12-31 23:00:00'
num_days: 6
chunk_plan:
  time: 144
  x: 175
  y: 175
queue: cpu
account: mappnat
interface: ib0
cores_per_job: 2
processes: 2
memory_per_job: 48
max_jobs: 12
walltime: '00:40:00'
```
| Option             | Description                                                                     |
|--------------------|---------------------------------------------------------------------------------| 
| num_chunks_per_job | The number of chunks to process per job-array job                               |
| wrf_dir            | The location of the source netCDF data files                                    |
| wrf_file_pat       | The naming pattern for the source data files                                    |
| target_dir         | Location to store the individual chunks of processed data                       |
| target_pat         | filename pattern to use for target stores                                       |
| temp_dir           | Location to store temporary files created during the rechunking process         |
| metadata_file      | Location of the metadata file to use for adding and updating variable metadata  |
| vars_file          | Location of file that lists the variables to process                            |
| dst_zarr           | Location of the final destination zarr dataset                                  |
| base_date          | The units used for the source time variable                                     |
| end_date           | The last date to use when creating the destination zarr store                   |
| num_days           | The number of days per chunk                                                    |
| chunk_plan         | The desired chunks to use for rechunking                                        |
| queue              | Name of job queue to use for job submissions                                    |
| account            | Account to use for jobs                                                         |
| interface          | Network interface to use (should be ib0)                                        |
| cores_per_job      | How many cpu cores to request per job                                           |
| processes          | Number of process to request                                                    |
| memory_per_job     | How much memory to request per core                                             |
| max_jobs           | Maximum number of job arrays to run at a time                                   |
| walltime           | Amount of walltime to request per job                                           |

## Testing
### Start interactive job
```bash
srun --pty -p cpu -A mappnat --ntasks=1 --cpus-per-task=128 --exclusive -t 08:00:00 -u bash -i
```

### Create zarr store
This will create an empty zarr dataset based on the given chunk index. The starting date for the zarr dataset is pulled from this chunk so typically you would always use the first chunk of model output for this process.
```bash
c404-ba_hourly_workflow create-zarr --config-file conus404-ba_hourly.yml --chunk-index=0
```
### Process source data in chunks to zarr store
The `num_chunks_per_job` variable in the configuration file controls how many chunks of model output are processed during the execution of the workflow. If this variable were set to 2 then the `chunk_index` argument would be in steps of 2 (e.g. 0, 2, 4, ...).
```bash
c404-ba_hourly_workflow process-wrf --config-file conus404-ba_hourly.yml --chunk-index=0
```
### Extend the time dimension in an existing zarr dataset
For the hourly workflow the `end_date` variable in the YAML file is used to indicate the date to extend the dataset to. This process is quite fast because it only updates the metadata for the zarr dataset. After the time dimension is extended then the source data chunks can be processed to fill in the values for the new time entries.
```bash
c404-ba_hourly_workflow extend-time --config-file conus404-ba_hourly.yml
```
