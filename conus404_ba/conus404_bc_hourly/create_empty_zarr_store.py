#!/usr/bin/env python3

import dask
import datetime
import fsspec
import json
import pandas as pd
import pyproj
import time
import warnings
import xarray as xr
import zarr

from cyclopts import App, Parameter, validators
from pathlib import Path
from zarr.convenience import consolidate_metadata
from zarr.util import NumberEncoder

from numcodecs import Zstd
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

from rich.console import Console
from rich import pretty

from typing import Annotated, Dict, List, Union

from ..conus404_helpers import get_accum_types, read_metadata
from ..conus404_config import Cfg

from .model_output_to_zarr import load_wrf_files, rechunk_job, resolve_path

warnings.simplefilter(action='ignore', category=FutureWarning)

pretty.install()
con = Console()

app = App(default_parameter=Parameter(negative=()))


def create_empty_zarr(src_zarr: Annotated[Path, Parameter(validator=validators.Path(exists=True))],
                      dst_zarr: Annotated[Path, Parameter(validator=validators.Path(exists=True))],
                      end_date: Union[datetime.datetime, datetime.date],
                      chunk_plan: Dict[str, int]):
    """Create an empty zarr store given a source zarr dataset

    :param src_zarr: source zarr dataset
    :param dst_zarr: destination zarr store to create
    :param end_date: last date to write in the destination zarr
    :param chunk_plan: dictionary containing chunk sizes
    """

    start_time = time.time()

    con.print('--- Create zarr store ---')
    ds = xr.open_dataset(src_zarr, engine='zarr', backend_kwargs=dict(consolidated=True), chunks={})

    dst_chunks = dict(y=chunk_plan['y'], x=chunk_plan['x'])
    time_chunk = chunk_plan['time']

    # Get integration information
    accum_types = get_accum_types(ds)
    drop_vars = accum_types.get('constant', [])

    # Get the full date range from the hourly zarr store
    # dates = pd.date_range(start=ds.time[0].values, end=ds.time[-1].values, freq='1h')
    dates = pd.date_range(start=ds.time[0].values, end=end_date, freq='1h')
    con.print(f'    date range: {ds.time.dt.strftime("%Y-%m-%d %H:%M:%S")[0].values} to '
              f'{dates[-1].strftime("%Y-%m-%d %H:%M:%S")}')
    con.print(f'    number of timesteps: {len(dates)}')

    # Get all variables but the constant variables
    source_dataset = ds.drop_vars(drop_vars, errors='ignore')

    # print('    --- Create template', end=' ')
    template = (source_dataset.chunk(dst_chunks).pipe(xr.zeros_like).isel(time=0, drop=True).expand_dims(time=len(dates)))
    template['time'] = dates
    template = template.chunk({'time': time_chunk})
    con.print(f'Create template:  {time.time() - start_time:0.3f} s')

    # print('    --- Write template', flush=True, end=' ')
    # Writes no data (yet)
    template.to_zarr(dst_zarr, compute=False, consolidated=True, mode='w')
    con.print(f'Write template: {time.time() - start_time:0.3f} s')

    # Remove the existing chunk encoding for constant variables
    for vv in drop_vars:
        try:
            del ds[vv].encoding['chunks']
        except KeyError:
            pass

    # Add the constant variables
    if len(drop_vars) > 0:
        ds[drop_vars].chunk(dst_chunks).to_zarr(dst_zarr, mode='a')
    con.print(f'Write constant variabls: {time.time() - start_time:0.3f} s')


def fix_crs(dst_zarr: Annotated[Path, Parameter(validator=validators.Path(exists=True))]):
    """Fix the crs variable and attributes in the zarr store

    :param dst_zarr: Path to the destination zarr store
    """

    # The consolidated metadata file for the zarr dataset
    zmetadata_file = f'{dst_zarr}/.zmetadata'

    ds = xr.open_dataset(dst_zarr, engine='zarr',
                         backend_kwargs=dict(consolidated=True),
                         decode_coords=True,
                         chunks={})

    # Convert the existing crs to a proper CRS and then convert
    # back to a cf-compliant set of attributes
    new_crs_attrs = pyproj.CRS(pyproj.CRS.from_cf(ds.crs.attrs)).to_cf()

    fs = fsspec.filesystem('file')

    with fs.open(zmetadata_file, 'r') as in_hdl:
        src_metadata = json.load(in_hdl)

    for kk, vv in src_metadata['metadata'].items():
        if kk in ['.zattrs', '.zgroup']:
            continue

        varname, metatype = kk.split('/')

        if metatype == '.zattrs':
            # Open the unconsolidated .zattrs file for the variable
            with fs.open(f'{dst_zarr}/{kk}', 'r') as in_hdl:
                orig_metadata = json.load(in_hdl)

            if varname == 'crs':
                # Completely replace the crs attributes but keep _ARRAY_DIMENSIONS
                zdim = orig_metadata['_ARRAY_DIMENSIONS']
                orig_metadata = new_crs_attrs
                orig_metadata['_ARRAY_DIMENSIONS'] = zdim

                # Change the datatype from |S1 to integer ('<i4') in the variable .zarray file
                with fs.open(f'{dst_zarr}/{varname}/.zarray', 'r') as in_hdl:
                    orig_zarray = json.load(in_hdl)
                orig_zarray['dtype'] = '<i4'

                with fs.open(f'{dst_zarr}/{varname}/.zarray', 'w') as out_hdl:
                    json.dump(orig_zarray, out_hdl, indent=4, sort_keys=True, ensure_ascii=True,
                              separators=(',', ': '), cls=NumberEncoder)

            # Write the updated metadata to the variable .zattrs file
            cfilename = f'{dst_zarr}/{kk}'
            with fs.open(cfilename, 'w') as out_hdl:
                json.dump(orig_metadata, out_hdl, indent=4, sort_keys=True, ensure_ascii=True,
                          separators=(',', ': '), cls=NumberEncoder)

    # # Now reopen the zarr dataset with unconsolidated metadata
    # ds = xr.open_dataset(fs.get_mapper(dst_zarr), engine='zarr',
    #                      backend_kwargs=dict(consolidated=False),
    #                      decode_coords=True, chunks={})

    # Write the new consolidated metadata
    consolidate_metadata(store=fs.get_mapper(dst_zarr), metadata_key='.zmetadata')


@app.default()
def run_job(config_file: str,
            chunk_index: int):

    job_name = f'wrf_rechunk_{chunk_index}'

    config = Cfg(config_file)

    wrf_dir = resolve_path('wrf_dir', config.wrf_dir)
    metadata_file = resolve_path('metadata_file', config.metadata_file)
    vars_file = resolve_path('vars_file', config.vars_file)
    dst_zarr = Path(config.dst_zarr).resolve()

    temp_dir = Path(config.temp_dir) / f'tmp_{chunk_index:05d}'

    con.print(f'{wrf_dir=}')
    con.print(f'{temp_dir=}')
    con.print(f'{dst_zarr=}')
    con.print(f'{vars_file=}')
    con.print('-'*60)

    chunk_plan = config.chunk_plan
    num_days = config.num_days
    base_date = config.base_date
    base_date = datetime.datetime.strptime(base_date, '%Y-%m-%d %H:%M:%S')
    delta = datetime.timedelta(days=num_days)

    con.print(base_date)

    con.print(f'{chunk_index=}')
    con.print(f'base_date={base_date.strftime("%Y-%m-%d %H:%M:%S")}')
    con.print(f'{num_days=}')
    con.print('-'*60)
    con.print(f'{chunk_plan=}')
    con.print('-'*60)

    # cluster = PBSCluster(job_name=job_name,
    #                      queue=config.queue,
    #                      account="NRAL0017",
    #                      interface='ib0',
    #                      cores=config.cores_per_job,
    #                      memory=config.memory_per_job,
    #                      walltime="05:00:00",
    #                      death_timeout=75)

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

    max_mem = f'{(config.memory_per_job / config.cores_per_job) * 0.8:0.1f}GB'
    # con.print(f'Maximum memory per thread for rechunking: {max_mem}')

    # Change the default compressor to Zstd
    zarr.storage.default_compressor = Zstd(level=9)

    fs = fsspec.filesystem('file')

    # Read variables to process
    df_vars = pd.read_csv(vars_file)
    var_list = df_vars['variable'].to_list()
    # var_list.append('time')
    con.print(f'Number of variables to process: {len(var_list)}')

    # Read the metadata file for modifications to variable attributes
    var_metadata = read_metadata(metadata_file)

    # Set the target directory
    target_dir = Path(config.target_dir) / f'{config.target_pat}{chunk_index:05d}'
    con.print(f'{target_dir=}')

    # Start date is selected based on chunk index
    st_date = base_date + datetime.timedelta(days=num_days * chunk_index)
    en_date = st_date + delta - datetime.timedelta(hours=1)

    con.print(f'{chunk_index}: {st_date.strftime("%Y-%m-%d %H:%M:%S")} to ' +
              f'{en_date.strftime("%Y-%m-%d %H:%M:%S")}')

    if (st_date - base_date).days % num_days != 0:
        con.print(f'[red]ERROR[/]: Start date must begin at the start of a {num_days}-day chunk')

    start_time = time.time()
    ds_wrf = load_wrf_files(num_days=num_days,
                            st_date=st_date,
                            file_pat=config.wrf_file_pat,
                            wrf_dir=wrf_dir)

    rechunk_job(chunk_index=chunk_index,
                max_mem=max_mem,
                ds_wrf=ds_wrf,
                target_dir=target_dir,
                temp_dir=temp_dir,
                var_metadata=var_metadata,
                var_list=var_list,
                chunk_plan=chunk_plan)

    end_time = time.time()
    con.print(f'{chunk_index}, elapsed time: {(end_time - start_time) / 60.:0.3f} minutes')

    # Create the empty final zarr destination
    create_empty_zarr(src_zarr=target_dir,
                      dst_zarr=dst_zarr,
                      end_date=config.end_date,
                      chunk_plan=chunk_plan)

    cluster.scale(0)


def main():
    app()


if __name__ == '__main__':
    main()
