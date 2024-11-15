# conus404_ba_hourly

The original conus404-ba workflow in 2023 used:
- `conus404_rechunk_bias_corrected.py`
- `conus404_biascorrected_to_zarr.py`

An updated workflow, `c404_ba_hourly_workflow.py`, was created in 2024 that includes workflow
commands to create an empty hourly zarr dataset, rechunk the bias-corrected data, and write the
rechunked data to the hourly zarr dataset.