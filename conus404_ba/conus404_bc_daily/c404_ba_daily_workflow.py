#!/usr/bin/env python3

from cyclopts import App, Parameter, validators
from pathlib import Path

import dask
# import datetime
# import fsspec
# import json
# import numpy as np
import pandas as pd
# import pyproj
import time
import warnings
import xarray as xr
import zarr
import zarr.storage

from rich.console import Console
from rich import pretty

from numcodecs import Zstd
from dask_jobqueue import SLURMCluster
from dask.distributed import Client   # , as_completed

# from typing import Annotated, Dict, List, Optional, Union
# from zarr.convenience import consolidate_metadata
# from zarr.util import NumberEncoder

from ..conus404_helpers import (apply_metadata, build_hourly_filelist, delete_dir, get_accum_types, read_metadata,
                                rechunker_wrapper)
from ..conus404_config import Cfg

warnings.simplefilter(action='ignore', category=FutureWarning)

pretty.install()
con = Console()

app = App(default_parameter=Parameter(negative=()))

var_attrs = dict(T2MAX=dict(coordinates='x y',
                            grid_mapping='crs',
                            long_name='Daily maximum temperature at 2 meters',
                            units='K'),
                 T2MIN=dict(coordinates='x y',
                            grid_mapping='crs',
                            long_name='Daily minimum temperature at 2 meters',
                            units='K'),
                 RAIN=dict(coordinates='x y',
                           grid_mapping='crs',
                           long_name='Daily accumulated precipitation',
                           standard_name='precipitation',
                           units='mm',
                           integration_length='24-hour accumulation'))


@app.command()
def create_zarr(config_file: str):
    config = Cfg(config_file)

    job_name = f'create_zarr'

    dask.config.set({"array.slicing.split_large_chunks": False})

    cluster = SLURMCluster(job_name=job_name,
                           queue=config.queue,
                           account=config.account,
                           interface=config.interface,
                           cores=config.cores_per_job,    # this is --cpus-per-task
                           processes=config.processes,    # this is numbers of workers for dask
                           memory=f'{config.memory_per_job} GiB',   # total amount of memory for job
                           walltime=config.walltime)

    con.print(cluster.job_script())
    cluster.scale(jobs=config.max_jobs)

    client = Client(cluster)
    client.wait_for_workers(config.processes * config.max_jobs)

    zarr.storage.default_compressor = Zstd(level=9)

    start_time = time.time()

    ds = xr.open_dataset(config.src_zarr, engine='zarr',
                         backend_kwargs=dict(consolidated=True), chunks={})

    ds['T2MAX'] = ds['T2D']
    ds['T2MIN'] = ds['T2D']
    ds['RAIN'] = ds['RAINRATE']

    for cvar in ds.variables:
        # Remove unneeded attributes, update the coordinates attribute

        # Add/modify attributes for current variable
        if cvar in var_attrs:
            for kk, vv in var_attrs[cvar].items():
                ds[cvar].attrs[kk] = vv

    # Get integration information
    accum_types = get_accum_types(ds)
    drop_vars = accum_types['constant']
    drop_vars.append('T2D')
    drop_vars.append('RAINRATE')

    # Get the full date range from the hourly zarr store
    dates = pd.date_range(start=ds.time[0].values, end=ds.time[-1].values, freq='1d')
    con.print(f'    date range: {ds.time.dt.strftime("%Y-%m-%d %H:%M:%S")[0].values} to '
              f'{dates[-1].strftime("%Y-%m-%d %H:%M:%S")}')
    con.print(f'    number of timesteps: {len(dates)}')

    # Get all variables but the constant variables
    source_dataset = ds.drop_vars(drop_vars, errors='ignore')

    dst_chunks = dict(y=config.chunk_plan['y'], x=config.chunk_plan['x'])
    time_chunk = config.chunk_plan['time']

    template = (source_dataset.chunk(dst_chunks).pipe(xr.zeros_like).isel(time=0,
                                                                          drop=True).expand_dims(time=len(dates)))
    template['time'] = dates
    template = template.chunk({'time': time_chunk})
    con.print(f'Create template:  {time.time() - start_time:0.3f} s')

    # Writes no data (yet)
    template.to_zarr(config.dst_zarr, compute=False, consolidated=True, mode='w')
    con.print(f'Write template: {time.time() - start_time:0.3f} s')

    # Remove the existing chunk encoding for constant variables
    for vv in drop_vars:
        try:
            del ds[vv].encoding['chunks']
        except KeyError:
            pass

    # Add the constant variables
    drop_vars.remove('T2D')
    drop_vars.remove('RAINRATE')

    ds[drop_vars].chunk(config.dst_chunks).to_zarr(config.dst_zarr, mode='a')
    con.print(f'Write constant variabls: {time.time() - start_time:0.3f} s')


def main():
    app()


if __name__ == '__main__':
    main()
