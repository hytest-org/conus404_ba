#!/usr/bin/env python3

from cyclopts import App, Parameter, Token, validators
from pathlib import Path

import dask
import datetime
import fsspec
import json
import math
import numpy as np
import os
import pandas as pd
import platform
import pyproj
import sys
import time
import warnings
import xarray as xr
import zarr
import zarr.storage

from rich.console import Console
from rich import pretty
from rich.padding import Padding
from rich.table import Table
from rich.progress import track

from numcodecs import Zstd
from dask_jobqueue import SLURMCluster
from dask.distributed import Client   # , as_completed

from typing import Annotated, Dict, List, Optional, Sequence, Union
from zarr.convenience import consolidate_metadata
from zarr.util import NumberEncoder

from ..version import __version__
from ..conus404_helpers import (apply_metadata, build_hourly_filelist, delete_dir, get_accum_types, read_metadata,
                                rechunker_wrapper)
from ..conus404_config import Cfg

warnings.simplefilter(action='ignore', category=FutureWarning)

pretty.install()
con = Console(record=True)

app = App(default_parameter=Parameter(negative=()))


def chunk_index_to_datetime_range(chunk_index: int,
                                  chunk_plan: Dict[str, int],
                                  base_date: Union[datetime.datetime, datetime.date]):
    """Given a chunk index, return the start and end date of the chunk.

    :param chunk_index: Index of the chunk
    :param chunk_plan: Dictionary of chunk sizes
    :param base_date: Base date of the time

    :returns: Tuple of start and end date of the chunk
    """

    num_days = int(chunk_plan['time']) / 24  # number of days per chunk
    delta = datetime.timedelta(days=num_days)

    st_date = base_date + datetime.timedelta(days=(num_days * chunk_index))
    en_date = st_date + delta - datetime.timedelta(hours=1)

    return st_date, en_date


def conv_datetime(type_, tokens: Sequence[Token]) -> datetime.datetime:
    """Convert datetime string from commandline to datetime object

    :param type_: type of object to convert to
    :param tokens: tokens to convert

    :return: datetime object
    """
    date_formats = ['%Y-%m-%d:%H',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H',
                    '%Y-%m-%dT%H:%M:%S']

    for cfmt in date_formats:
        try:
            return datetime.datetime.strptime(tokens[0].value, cfmt)
        except ValueError:
            continue
    raise ValueError


def datetime64_to_datetime(dt64: np.datetime64) -> datetime.datetime:
    """Convert a numpy datetime64 object to a datetime object.

    :param dt64: Numpy datetime64 object

    :returns: Datetime object
    """

    # Derived from: https://www.delftstack.com/howto/numpy/numpy-convert-datetime64/
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    seconds_since_epoch = (dt64 - unix_epoch) / one_second

    return datetime.datetime.utcfromtimestamp(seconds_since_epoch)


def datetime_to_chunk_index(check_date: Union[datetime.datetime, datetime.date],
                            chunk_plan: Dict[str, int],
                            base_date: Union[datetime.datetime, datetime.date]) -> int:
    """Given a date, return the chunk index.

    :param check_date: Date to check
    :param chunk_plan: Dictionary of chunk sizes
    :param base_date: Base date of the time

    :returns: Index of the time chunk
    """

    num_days = chunk_plan['time'] / 24  # number of days per chunk
    delta = datetime.timedelta(days=num_days)

    chunk_index = math.floor((check_date - base_date) / delta)
    return chunk_index


def _get_base_date(da: xr.DataArray) -> datetime.datetime:
    """Get the base date for a time data array

    :param da: time data array

    :return: base date
    """

    return datetime.datetime.strptime(da.encoding['units'].removeprefix('hours since '), '%Y-%m-%d %H:%M:%S')


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

    if 'crs' in drop_vars:
        crs_ds = create_crs(ds)
        crs_ds.to_zarr(dst_zarr, mode='a')
        drop_vars.remove('crs')

    # Add the constant variables
    if len(drop_vars) > 0:
        ds[drop_vars].chunk(dst_chunks).to_zarr(dst_zarr, mode='a')
    con.print(f'Write constant variabls: {time.time() - start_time:0.3f} s')


def create_crs(ds: xr.Dataset) -> xr.Dataset:
    """Create a dataset with new crs variable derived from the source dataset crs

    :param ds: source dataset
    :return: dataset with new crs variable
    """

    # Convert the existing crs to a proper CRS and then convert
    # back to a cf-compliant set of attributes
    new_crs_attrs = pyproj.CRS(pyproj.CRS.from_cf(ds.crs.attrs)).to_cf()

    # Create a dataset with the CRS
    dsn = xr.Dataset(data_vars={'crs': ([], 1, new_crs_attrs)})

    return dsn


def load_wrf_files(num_days: int,
                   st_date: Union[datetime.datetime, datetime.date],
                   file_pat: str,
                   wrf_dir: Annotated[Path, Parameter(validator=validators.Path(exists=True))]) -> xr.Dataset:
    """Load WRF model output netCDF files into an xarray dataset

    :param num_days: number of days of hourly files to load
    :param st_date: start date for the chunk
    :param file_pat: file pattern for WRF model output files
    :param wrf_dir: directory containing WRF model output files
    :return: xarray dataset containing WRF model output
    """

    concat_dim = 'time'
    try:
        job_files = build_hourly_filelist(num_days, st_date, wrf_dir, file_pat, verify=False)
        ds = xr.open_mfdataset(job_files, concat_dim=concat_dim, combine='nested',
                               parallel=True, coords="minimal", data_vars="minimal",
                               engine='netcdf4', compat='override', chunks={})
    except FileNotFoundError:
        # Re-run the filelist build with the expensive verify
        job_files = build_hourly_filelist(num_days, st_date, wrf_dir, file_pat, verify=True)
        con.print(job_files[0])
        con.print(job_files[-1])
        con.print(f'Number of valid files: {len(job_files)}')

        ds = xr.open_mfdataset(job_files, concat_dim=concat_dim, combine='nested',
                               parallel=True, coords="minimal", data_vars="minimal",
                               engine='netcdf4', compat='override', chunks={})

    return ds


def ptext(txt: str,
          style: str = 'none',
          max_width: int = 40,
          expand: bool = False) -> Padding:
    """Returns a padded text object

    :param txt: Text to print
    :param style: Style for padding characters (e.g. bold)
    :param max_width: Maximum width of the line
    :param expand: Expand padding to fit a available width
    :return: Padded
    """

    lr_pad = int((max_width - len(txt)) / 2)
    return Padding(txt, (0, lr_pad), style=style, expand=expand)


def rechunk_job(chunk_index: int,
                max_mem: Union[float, str],
                ds_wrf: xr.Dataset,
                target_dir: Annotated[Path, Parameter(validator=validators.Path(exists=True))],
                temp_dir: Annotated[Path, Parameter(validator=validators.Path(exists=True))],
                var_metadata: pd.DataFrame,
                var_list: List[str],
                chunk_plan: Dict[str, int]):
    """Use rechunker to rechunk WRF netcdf model output files into zarr format

    :param chunk_index: index of chunk to process
    :param max_mem: maximum memory per thread
    :param ds_wrf:
    :param target_dir: directory for target zarr files
    :param temp_dir: directory for temporary zarr files during rechunking
    :param var_metadata: dataframe containing variable metadata
    :param var_list: list of variables to process
    :param chunk_plan: dictionary containing chunk sizes for rechunking
    """

    # Attributes that should be removed from all variables
    remove_attrs = ['esri_pe_string', 'proj4', '_CoordinateAxisType', 'resolution']

    fs = fsspec.filesystem('file')

    start_time = time.time()

    # Rechunker requires empty temp and target dirs
    delete_dir(fs, temp_dir)
    delete_dir(fs, target_dir)
    time.sleep(3)  # Wait for files to be removed (necessary? hack!)

    ds_wrf = apply_metadata(ds_wrf, {}, {}, remove_attrs, var_metadata)

    # with performance_report(filename=f'dask_perf_{args.index}.html'):
    rechunker_wrapper(ds_wrf[var_list], target_store=target_dir, temp_store=temp_dir,
                      mem=max_mem, consolidated=True, verbose=False,
                      chunks=chunk_plan)

    end_time = time.time()
    con.print(f'  {chunk_index}: rechunker elapsed time: {(end_time - start_time) / 60.:0.3f} minutes')


def to_zarr(chunk_index: int,
            chunk_plan: Dict[str, int],
            src_zarr: Annotated[Path, Parameter(validator=validators.Path(exists=True))],
            dst_zarr: Annotated[Path, Parameter(validator=validators.Path(exists=True))]):
    """Insert chunk into existing zarr store

    :param chunk_index: index of chunk to insert
    :param chunk_plan: dictionary containing chunk sizes
    :param src_zarr: source zarr chunk
    :param dst_zarr: destination zarr store
    """

    # Get the time values from the destination zarr store
    ds_dst = xr.open_dataset(dst_zarr, engine='zarr', mask_and_scale=True, chunks={})
    dst_time = ds_dst.time.values

    start_time = time.time()

    # start = chunk_index * chunk_plan['time']
    # stop = (chunk_index + 1) * chunk_plan['time']

    try:
        ds_src = xr.open_dataset(src_zarr, engine='zarr', mask_and_scale=True, chunks={})
    except FileNotFoundError:
        con.print(f'[red]ERROR[/]: {src_zarr} does not exist; skipping.')
        return

    drop_vars = get_accum_types(ds_src).get('constant', [])

    st_date_src = ds_src.time.values[0]
    en_date_src = ds_src.time.values[-1]
    st_time_idx = np.where(dst_time == st_date_src)[0].item()
    en_time_idx = np.where(dst_time == en_date_src)[-1].item() + 1

    # con.print(f'  time slice: {start}, {stop} = {stop-start}')
    # con.print(f'  dst time slice: {st_time_idx}, {en_time_idx} = {en_time_idx - st_time_idx}')

    # Drop the constants
    ds_src = ds_src.drop_vars(drop_vars, errors='ignore')
    ds_src.to_zarr(dst_zarr, region={'time': slice(st_time_idx, en_time_idx)})

    end_time = time.time()
    con.print(f'  {chunk_index}: to_zarr elapsed time: {(end_time - start_time) / 60.:0.3f} minutes')


def resolve_path(msg: str, path: str) -> Path:
    """
    Resolve a path and exit if it does not exist

    :param msg: message to include in error message
    :param path: path to resolve
    :return: Resolved path
    """

    try:
        path = Path(path).resolve(strict=True)
    except FileNotFoundError:
        con.print(f'[red]{msg}[/]: {path} does not exist')
        exit()

    return path


@app.command()
def check_min_max(config_file: str,
                  variables: Annotated[list[str], Parameter(consume_multiple=True)]):
    """Get minimum and maximum values from dataset for selected variables

    :param config_file: Name of configuration file
    :param variables: List of variables to process
    """

    start_time_pgm = time.time()
    job_name = f'check_range'

    config = Cfg(config_file)

    dst_zarr = resolve_path('dst_zarr', config.dst_zarr)
    ds = xr.open_dataset(dst_zarr, engine='zarr', backend_kwargs=dict(consolidated=True), chunks={})

    con.print(ptext('Settings', style='white on tan', max_width=80, expand=False))
    con.print(f'[green4]INFO[/]: Start {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    con.print(f'[green4]INFO[/]: Command line: [grey39]{" ".join(sys.argv)}[/]')
    con.print(f'[green4]INFO[/]: Package version: {__version__}')
    con.print(f'[green4]INFO[/]: Script directory: [grey39]{os.path.dirname(os.path.abspath(__file__))}[/]')
    con.print(f'[green4]INFO[/]: Python: [grey39]{platform.python_implementation()} ({platform.python_version()})[/]')
    con.print(f'[green4]INFO[/]: Host: [grey39]{platform.node()}[/]')
    con.print('-'*80)

    con.print(f'[green4]INFO[/]: SLURM_NODENAME: {os.environ.get("SLURM_NODENAME", "")}')
    con.print(f'[green4]INFO[/]: SLURM_ARRAY_TASK_ID: {os.environ.get("SLURM_ARRAY_TASK_ID", "")}')
    con.print(f'[green4]INFO[/]: SLURM_NODE_LIST: {os.environ.get("SLURM_NODE_LIST", "")}')
    con.print(f'[green4]INFO[/]: Dask temporary directory: {dask.config.get("temporary-directory")}')
    con.print('-'*80)
    con.print(f'[green4]INFO[/]: Destination hourly dataset: {config.dst_zarr}')
    con.print('='*80)

    cluster = SLURMCluster(job_name=job_name,
                           queue=config.queue,
                           account=config.account,
                           interface=config.interface,
                           cores=config.cores_per_job,    # this is --cpus-per-task
                           processes=config.processes,    # this is numbers of workers for dask
                           memory=f'{config.memory_per_job} GiB',   # total amount of memory for job
                           walltime=config.walltime,
                           job_extra_directives=['--exclusive'])

    con.print(cluster.job_script())
    cluster.scale(jobs=config.max_jobs)

    client = Client(cluster)
    client.wait_for_workers(config.processes * config.max_jobs)

    st_yr = ds['time'].dt.year[0].item()
    en_yr = ds['time'].dt.year[-1].item()

    con.print(f'Minimum and maximum values for {st_yr} to {en_yr}')
    for cvar in variables:
        o_max = -2000000.0
        o_min = 2000000.0

        for cyr in range(st_yr, en_yr+1):
            date_range = slice(f'{cyr}-01-01', f'{cyr}-12-31')
            ds_sub = ds[cvar].sel(time=date_range)
            min_val, max_val = dask.compute(ds_sub.min(skipna=False), ds_sub.max(skipna=False))
            min_val = min_val.item()
            max_val = max_val.item()
            o_max = max(max_val, o_max)
            o_min = min(min_val, o_min)

            con.print(f'{cvar} - {cyr}: {min_val:0.4f} to {max_val:0.4f}')
        con.print(f'[bold]{cvar}[/]: {o_min}, {o_max}')
    cluster.scale(0)

    con.print(f'[green4]INFO[/]: Total program runtime: {(time.time() - start_time_pgm) / 60.:0.3f} m')


@app.command()
def check_missing(config_file: str,
                  variables: Annotated[list[str], Parameter(consume_multiple=True)]):
    """Check dataset for missing values

    :param config_file: Name of configuration file
    :param variables: List of variables to process
    """

    # con.print(f'HOST: {os.environ.get("HOSTNAME")}')
    # con.print(f'SLURMD_NODENAME: {os.environ.get("SLURMD_NODENAME")}')

    config = Cfg(config_file)

    # con.print(f'dask tmp directory: {dask.config.get("temporary-directory")}')

    start_time = time.time()
    workers = 12
    threads_per = 2
    dask.config.set({"array.slicing.split_large_chunks": False})

    con.print(ptext('Settings', style='white on tan', max_width=80, expand=False))
    # start_time = time.time()
    con.print(f'[green4]INFO[/]: Start {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    con.print(f'[green4]INFO[/]: Command line: [grey39]{" ".join(sys.argv)}[/]')
    con.print(f'[green4]INFO[/]: Package version: {__version__}')
    con.print(f'[green4]INFO[/]: Script directory: [grey39]{os.path.dirname(os.path.abspath(__file__))}[/]')
    con.print(f'[green4]INFO[/]: Python: [grey39]{platform.python_implementation()} ({platform.python_version()})[/]')
    con.print(f'[green4]INFO[/]: Host: [grey39]{platform.node()}[/]')
    con.print('-'*80)

    con.print(f'[green4]INFO[/]: SLURM_NODENAME: {os.environ.get("SLURM_NODENAME", "")}')
    con.print(f'[green4]INFO[/]: SLURM_ARRAY_TASK_ID: {os.environ.get("SLURM_ARRAY_TASK_ID", "")}')
    con.print(f'[green4]INFO[/]: SLURM_NODE_LIST: {os.environ.get("SLURM_NODE_LIST", "")}')
    con.print(f'[green4]INFO[/]: Dask temporary directory: {dask.config.get("temporary-directory")}')
    con.print(f'[green4]INFO[/]: Hourly dataset: {config.dst_zarr}')
    con.print('='*80)

    job_name = 'check_missing'

    cluster = SLURMCluster(job_name=job_name,
                           queue=config.queue,
                           account=config.account,
                           interface=config.interface,
                           cores=config.cores_per_job,    # this is --cpus-per-task
                           processes=config.processes,    # this is numbers of workers for dask
                           memory=f'{config.memory_per_job} GiB',   # total amount of memory for job
                           walltime=config.walltime,
                           job_extra_directives=['--exclusive'])

    con.print(cluster.job_script())
    cluster.scale(jobs=config.max_jobs)

    client = Client(cluster)
    client.wait_for_workers(config.processes * config.max_jobs)

    # with Client(n_workers=workers, threads_per_worker=threads_per, memory_limit=None):
    ds = xr.open_dataset(config.dst_zarr, engine='zarr', backend_kwargs=dict(consolidated=True), chunks={})

    st_yr = ds['time'].dt.year[0].item()
    en_yr = ds['time'].dt.year[-1].item()

    # varlist = ['RAINRATE', 'T2D', 'LWDOWN', 'PSFC', 'Q2D', 'SWDOWN', 'U2D', 'V2D']
    varlist = list(ds.data_vars)
    if 'crs' in varlist:
        varlist.remove('crs')

    con.print(f'Checking for missing values in {len(variables)} variables: {variables}')
    con.print(f'[green4]INFO[/]: Year range: {st_yr} to {en_yr}')

    proc_info = Table(title='[bold]Years with missing values[/]')
    proc_info.add_column('Variable', justify='left', style='black')
    proc_info.add_column('Year', justify='left', style='black')
    proc_info.add_column('Max missing', justify='left', style='red')
    # proc_info.add_column('Index range', justify='left', style='grey50')

    for cvar in varlist:
        for cyear in track(range(st_yr, en_yr + 1), description=f'Processing {cvar}'):
            ds1 = ds[cvar].sel(time=slice(f'{cyear}-01-01', f'{cyear}-12-31'))

            non_nan_count = ds1.count(dim='time').compute()
            total_timesteps = ds1.sizes['time']

            nan_count = total_timesteps - non_nan_count.astype(int)

            max_missing = nan_count.where(nan_count > 0).max().item()
            if not np.isnan(max_missing):
                proc_info.add_row(cvar, str(cyear), str(max_missing))
                # print(f'{cyear}: Max missing values: {max_missing}')

    if len(proc_info.rows) > 0:
        con.print(proc_info)
    else:
        con.print('[green4]INFO[/]: No variables contained missing values')

    cluster.scale(0)
    con.save_html(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M")}_conus404-pgw-bc_check_missing.html')


@app.command()
def create_zarr(config_file: str, chunk_index: int):
    """Create a zarr store using a single chunk of WRF model output files

    :param config_file: Name of configuration file
    :param chunk_index: index of chunk to insert
    """

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


@app.command()
def extend_time(config_file: str,
                freq: Optional[str] = '1h'):
    """Extend the time dimension in an existing zarr dataset

    :param config_file: Name of configuration file
    :param freq: frequency to use for timesteps
    """

    config = Cfg(config_file)

    dst_zarr = Path(config.dst_zarr).resolve()
    end_date = config.end_date
    dst_filename = f'{dst_zarr}/.zmetadata'

    # Read the consolidated metadata
    with open(dst_filename, 'r') as in_hdl:
        data = json.load(in_hdl)

    # Open the target zarr dataset
    con.print('  reading zarr store')
    ds = xr.open_dataset(dst_zarr, engine='zarr',
                         backend_kwargs=dict(consolidated=True), chunks={})

    if pd.to_datetime(end_date) == ds.time[-1].values:
        con.print(f'  [green]INFO[/]: {end_date} is already the last date in the zarr dataset')
        return

    con.print(f'Zarr store: {dst_zarr}')
    con.print(f'Original end date: {ds.time[-1].values}')
    con.print(f'New end date: {end_date}')
    con.print('-'*40)

    # Define the new time range
    # Date range should always start from the original starting date in the zarr dataset
    dates = pd.date_range(start=ds.time[0].values, end=end_date, freq=freq)

    con.print('  reading metadata')
    # Get the index for time dimension of each variable from the consolidated metadata
    time_index = {}

    for kk, vv in data['metadata'].items():
        if kk in ['.zattrs', '.zgroup']:
            continue

        varname, metatype = kk.split('/')

        if metatype == '.zattrs':
            try:
                time_index[varname] = vv['_ARRAY_DIMENSIONS'].index('time')
            except ValueError:
                # Time dimension not used for this variable
                pass

            # con.print(f'{kk} {vv["_ARRAY_DIMENSIONS"]}')

    # Index for the time dimension for each variable
    # con.print(time_index)

    # Change the size of the time dimension in the unconsolidated metadata for each variable
    # This will overwrite the original .zarray file for each variable
    con.print('  updating metadata')
    for kk, vv in time_index.items():
        cfilename = f'{dst_zarr}/{kk}/.zarray'

        with open(cfilename, 'r') as in_hdl:
            uncol_meta = json.load(in_hdl)

        # Update the shape of the variable
        uncol_meta['shape'][vv] = len(dates)

        # con.print('-'*10, kk, '-'*10)
        # con.print(uncol_meta)

        # Write the updated metadata file
        with open(cfilename, 'w') as out_hdl:
            json.dump(uncol_meta, out_hdl, indent=4, sort_keys=True, ensure_ascii=True,
                      separators=(',', ': '), cls=NumberEncoder)

    ds.close()

    con.print('  consolidating metadata')
    # Re-open the zarr datastore using the unconsolidated metadata
    ds = xr.open_dataset(dst_zarr, engine='zarr',
                         backend_kwargs=dict(consolidated=False), chunks={},
                         decode_times=False)

    # Write a new consolidated metadata file
    consolidate_metadata(store=dst_zarr, metadata_key='.zmetadata')

    # Write the new time values
    con.print('  updating time variable in zarr store')
    save_enc = ds['time'].encoding

    # We opened the dataset without decoding time so we need to also
    # grab the `units` and `calendar` attributes to add to the encoding
    save_attrs = ds['time'].attrs

    ds.coords['time'] = dates
    ds['time'].encoding.update(save_enc)

    for kk, vv in save_attrs.items():
        if kk in ['units', 'calendar']:
            ds['time'].encoding[kk] = vv

    ds[['time']].to_zarr(dst_zarr, mode='a')


@app.command()
def get_chunk_index(check_date: Annotated[datetime.datetime, Parameter(converter=conv_datetime)],
                    config_file: str):
    """Output the chunk index for a datetime value

    :param check_date: datetime value to check
    :param config_file: Name of configuration file
    """

    config = Cfg(config_file)

    # Open the destination zarr
    # ds = xr.open_dataset(config.dst_zarr, engine='zarr', backend_kwargs=dict(consolidated=True), chunks={})

    # Get the base_date and chunk_plan from the destination dataset
    base_date = datetime.datetime.strptime(config.base_date, '%Y-%m-%d %H:%M:%S')
    chunk_plan = config.chunk_plan

    chunk_index = datetime_to_chunk_index(check_date, chunk_plan, base_date)
    con.print(f'{check_date.strftime("%Y-%m-%dT%H")} is in chunk {chunk_index}')


@app.command()
def info(config_file: str):
    """Output information about the destination zarr store

    :param config_file: Name of configuration file
    """

    config = Cfg(config_file)

    # Open the destination zarr
    ds = xr.open_dataset(config.dst_zarr, engine='zarr', backend_kwargs=dict(consolidated=True), chunks={})

    # Get the base_date and chunk_plan from the destination dataset
    base_date = datetime.datetime.strptime(config.base_date, '%Y-%m-%d %H:%M:%S')
    chunk_plan = config.chunk_plan

    con.print(f'base_date: {base_date.strftime("%Y-%m-%d %H:%M:%S")}')
    con.print(f'{chunk_plan=}')

    first_date = datetime64_to_datetime(ds.time[0].values)
    last_date = datetime64_to_datetime(ds.time[-1].values)

    first_chunk = datetime_to_chunk_index(first_date, chunk_plan, base_date)
    last_chunk = datetime_to_chunk_index(last_date, chunk_plan, base_date)

    first_chunk_st, first_chunk_en = chunk_index_to_datetime_range(chunk_index=first_chunk,
                                                                   chunk_plan=chunk_plan, base_date=base_date)
    last_chunk_st, last_chunk_en = chunk_index_to_datetime_range(chunk_index=last_chunk,
                                                                 chunk_plan=chunk_plan, base_date=base_date)

    con.print(f'First date: {first_date.strftime("%Y-%m-%dT%H")}, '
              f'chunk: {first_chunk} ({first_chunk_st.strftime("%Y-%m-%dT%H")} to {first_chunk_en.strftime("%Y-%m-%dT%H")})')
    con.print(f'Last date: {last_date.strftime("%Y-%m-%dT%H")}, '
              f'chunk: {last_chunk} ({last_chunk_st.strftime("%Y-%m-%dT%H")} to {last_chunk_en.strftime("%Y-%m-%dT%H")})')

    if last_date < last_chunk_en:
        con.print(f'[green]INFO[/]: Last chunk in dataset is a partial chunk')


@app.command()
def process(config_file: str,
            chunk_index: int):
    """Rechunk WRF model output files and insert into existing zarr store

    :param config_file: Name of configuration file
    :param chunk_index: Index of the chunk to process
    """

    start_time_pgm = time.time()
    job_name = f'wrf_rechunk_{chunk_index}'

    config = Cfg(config_file)

    wrf_dir = resolve_path('wrf_dir', config.wrf_dir)
    metadata_file = resolve_path('metadata_file', config.metadata_file)
    vars_file = resolve_path('vars_file', config.vars_file)
    dst_zarr = resolve_path('dst_zarr', config.dst_zarr)

    # temp_dir = Path(config.temp_dir)

    # con.print(f'{wrf_dir=}')
    # con.print(f'{temp_dir=}')
    # con.print(f'{dst_zarr=}')
    # con.print(f'{metadata_file=}')
    # con.print(f'{vars_file=}')
    # con.print('-'*60)

    chunk_plan = config.chunk_plan
    num_days = config.num_days
    base_date = config.base_date
    base_date = datetime.datetime.strptime(base_date, '%Y-%m-%d %H:%M:%S')
    delta = datetime.timedelta(days=num_days)

    # con.print(base_date)
    #
    # con.print(f'{chunk_index=}')
    # con.print(f'base_date={base_date.strftime("%Y-%m-%d %H:%M:%S")}')
    # con.print(f'{num_days=}')
    # con.print('-'*60)
    # con.print(f'{chunk_plan=}')
    # con.print('-'*60)

    dask.config.set({"array.slicing.split_large_chunks": False})

    con.print(ptext('Settings', style='white on tan', max_width=80, expand=False))
    con.print(f'[green4]INFO[/]: Start {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    con.print(f'[green4]INFO[/]: Command line: [grey39]{" ".join(sys.argv)}[/]')
    con.print(f'[green4]INFO[/]: Package version: {__version__}')
    con.print(f'[green4]INFO[/]: Script directory: [grey39]{os.path.dirname(os.path.abspath(__file__))}[/]')
    con.print(f'[green4]INFO[/]: Python: [grey39]{platform.python_implementation()} ({platform.python_version()})[/]')
    con.print(f'[green4]INFO[/]: Host: [grey39]{platform.node()}[/]')
    con.print('-'*80)

    con.print(f'[green4]INFO[/]: SLURM_NODENAME: {os.environ.get("SLURM_NODENAME", "")}')
    con.print(f'[green4]INFO[/]: SLURM_ARRAY_TASK_ID: {os.environ.get("SLURM_ARRAY_TASK_ID", "")}')
    con.print(f'[green4]INFO[/]: SLURM_NODE_LIST: {os.environ.get("SLURM_NODE_LIST", "")}')
    con.print(f'[green4]INFO[/]: Dask temporary directory: {dask.config.get("temporary-directory")}')
    con.print('-'*80)
    con.print(f'[green4]INFO[/]: Variable list file: {vars_file}')
    con.print(f'[green4]INFO[/]: Variable metadata file: {metadata_file}')
    con.print(f'[green4]INFO[/]: Source WRF netCDF path: {config.wrf_dir}')
    con.print(f'[green4]INFO[/]: Base date: {base_date.strftime("%Y-%m-%d %H:%M:%S")}')
    con.print(f'[green4]INFO[/]: Chunk plan: {chunk_plan}')
    con.print(f'[green4]INFO[/]: Number of days per chunk: {num_days}')
    con.print(f'[green4]INFO[/]: Chunk index to start processing: {chunk_index}')
    con.print(f'[green4]INFO[/]: Number of chunks to process: {config.num_chunks_per_job}')
    con.print(f'[green4]INFO[/]: Rechunker temporary base path: {config.temp_dir}')
    con.print(f'[green4]INFO[/]: Rechunker target base path: {config.target_dir}')
    con.print(f'[green4]INFO[/]: Destination hourly dataset: {config.dst_zarr}')
    con.print('='*80)

    # cluster = PBSCluster(job_name=job_name,
    #                      queue=config.queue,
    #                      account="NRAL0017",
    #                      interface='ib0',
    #                      cores=config.cores_per_job,
    #                      memory=config.memory_per_job,
    #                      walltime="05:00:00",
    #                      death_timeout=75)

    cluster = SLURMCluster(job_name=job_name,
                           queue=config.queue,
                           account=config.account,
                           interface=config.interface,
                           cores=config.cores_per_job,    # this is --cpus-per-task
                           processes=config.processes,    # this is numbers of workers for dask
                           memory=f'{config.memory_per_job} GiB',   # total amount of memory for job
                           walltime=config.walltime)
                           # job_cpu=8,   # this appears to override cores, but cores is still a required argument
                           # job_extra_directives=['--nodes=6'],
                           # local_directory='/home/pnorton/hytest/hourly_processing')

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
    var_list.append('time')
    con.print(f'[green4]INFO[/]: Number of variables to process: {len(var_list)}')

    # Read the metadata file for modifications to variable attributes
    var_metadata = read_metadata(metadata_file)

    for cidx in range(chunk_index, chunk_index+config.num_chunks_per_job):
        # Set the target directory
        target_dir = Path(config.target_dir) / f'{config.target_pat}{cidx:05d}'
        temp_dir = Path(config.temp_dir) / f'tmp_{cidx:05d}'
        # con.print(f'{target_dir=}')
        # con.print(f'{temp_dir=}')

        # Start date is selected based on chunk index
        st_date = base_date + datetime.timedelta(days=num_days * cidx)
        en_date = st_date + delta - datetime.timedelta(hours=1)

        con.print(f'{cidx}: {st_date.strftime("%Y-%m-%d %H:%M:%S")} to ' +
                  f'{en_date.strftime("%Y-%m-%d %H:%M:%S")}')

        if (st_date - base_date).days % num_days != 0:
            con.print(f'[red]ERROR[/]: Start date must begin at the start of a {num_days}-day chunk')

        start_time = time.time()
        ds_wrf = load_wrf_files(num_days=num_days,
                                st_date=st_date,
                                file_pat=config.wrf_file_pat,
                                wrf_dir=wrf_dir)

        rechunk_job(chunk_index=cidx,
                    max_mem=max_mem,
                    ds_wrf=ds_wrf,
                    target_dir=target_dir,
                    temp_dir=temp_dir,
                    var_metadata=var_metadata,
                    var_list=var_list,
                    chunk_plan=chunk_plan)

        to_zarr(chunk_index=cidx,
                chunk_plan=chunk_plan,
                src_zarr=target_dir,
                dst_zarr=dst_zarr)

        # print('Removing target directory')
        delete_dir(fs, target_dir)
        delete_dir(fs, temp_dir)

        end_time = time.time()
        con.print(f'{cidx}: Total elapsed time: {(end_time - start_time) / 60.:0.3f} minutes')

    cluster.scale(0)

    con.print(f'[green4]INFO[/]: Total program runtime: {(time.time() - start_time_pgm) / 60.:0.3f} m')


def main():
    app()


if __name__ == '__main__':
    main()
