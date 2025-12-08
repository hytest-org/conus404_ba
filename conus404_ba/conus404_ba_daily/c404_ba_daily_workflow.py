#!/usr/bin/env python3

from cyclopts import App, Parameter, Token   # , validators
from pathlib import Path

import dask
import datetime
import json
import math
import numpy as np
import os
import pandas as pd
import platform
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
from ..conus404_helpers import get_accum_types
from ..conus404_config import Cfg

warnings.simplefilter(action='ignore', category=FutureWarning)

pretty.install()
con = Console(record=True)

app = App(default_parameter=Parameter(negative=()))


def adjust_time(ds: xr.Dataset, time_adj: int) -> xr.Dataset:
    """Align time boundaries after computing daily values from hourly

    :param ds: Dataset to adjust time values
    :param time_adj: Number of seconds to adjust
    :return: Dataset with adjust time values
    """

    # Adjust the time values, pass the original encoding to the new time index
    save_enc = ds.time.encoding
    del save_enc['chunks']

    ds['time'] = ds['time'] - np.timedelta64(time_adj, 'm')
    ds.time.encoding = save_enc

    return ds


def chunk_index_to_datetime_range(chunk_index: int,
                                  chunk_plan: Dict[str, int],
                                  base_date: Union[datetime.datetime, datetime.date]):
    """Given a chunk index, return the start and end date of the chunk.

    :param chunk_index: Index of the chunk
    :param chunk_plan: Dictionary of chunk sizes
    :param base_date: Base date of the time

    :returns: Tuple of start and end date of the chunk
    """

    num_days = int(chunk_plan['time'])   # number of days per chunk
    delta = datetime.timedelta(days=num_days)

    st_date = base_date + datetime.timedelta(days=(num_days * chunk_index))
    en_date = st_date + delta - datetime.timedelta(hours=24)

    return st_date, en_date


def compute_daily(ds: xr.Dataset,
                  var_list: List,
                  st_idx: int,
                  en_idx: int,
                  chunks: Optional[Dict[str, int]] = None) -> xr.Dataset:
    """Compute daily values from a source hourly dataset

    :param ds: Source hourly dataset
    :param var_list: List of variables to process
    :param st_idx: Starting index in source hourly dataset
    :param en_idx: Ending index in source hourly dataset
    :param chunks: Dictionary containing chunk sizes
    :return: dataset of daily values
    """

    if chunks is None:
        chunks = {}

    # NOTE: make sure the arrays have the correct final chunking.

    ds_day_cnk = ds[var_list].isel(time=slice(st_idx, en_idx))
    # con.print(f'    instant: hourly range: {st_idx} ({ds_day_cnk.time.dt.strftime("%Y-%m-%d %H:%M:%S")[0].values}) to '
    #           f'{en_idx} ({ds_day_cnk.time.dt.strftime("%Y-%m-%d %H:%M:%S")[-1].values})')

    if 'T2D' in var_list:
        ds_max = ds_day_cnk['T2D'].coarsen(time=24, boundary='pad').max(skipna=False).chunk(chunks)
        ds_max.name = 'T2MAX'

        ds_min = ds_day_cnk['T2D'].coarsen(time=24, boundary='pad').min(skipna=False).chunk(chunks)
        ds_min.name = 'T2MIN'

    if 'RAINRATE' in var_list:
        ds_rain = (ds_day_cnk['RAINRATE'] * 3600).coarsen(time=24, boundary='pad').sum(skipna=False).chunk(chunks)
        ds_rain.name = 'RAIN'

    if 'T2D' in var_list and 'RAINRATE' in var_list:
        ds_daily = xr.merge([ds_min, ds_max, ds_rain])
        return ds_daily
    elif 'T2D' in var_list:
        ds_daily = xr.merge([ds_min, ds_max])
        return ds_daily
    elif 'RAINRATE' in var_list:
        ds_daily = xr.merge([ds_rain])
        return ds_daily


def conv_datetime(type_, tokens: Sequence[Token]) -> datetime.datetime:
    """Convert datetime string from commandline to datetime object

    :param type_: type of object to convert to
    :param tokens: tokens to convert

    :return: datetime object
    """
    date_formats = ['%Y-%m-%d']

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

    num_days = chunk_plan['time']   # number of days per chunk
    delta = datetime.timedelta(days=num_days)

    chunk_index = math.floor((check_date - base_date) / delta)
    return chunk_index


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


def remove_chunk_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Remove existing encoding from variables in a dataset

    :param ds: Dataset to remove encoding from
    :return: Dataset with variable chunk encoding removed
    """

    # Remove the existing encoding for chunks
    for vv in ds.variables:
        try:
            del ds[vv].encoding['chunks']
        except KeyError:
            pass

    return ds


@app.command()
def check_min_max(config_file: str,
                  variables: Annotated[list[str], Parameter(consume_multiple=True)]):
    """Get minimum and maximum values from dataset for selected variables

    :param config_file: Name of configuration file
    :param variables: List of variables to process
    """

    start_time_pgm = time.time()
    job_name = f'wrf_check_range'

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
    con.print(f'[green4]INFO[/]: Daily dataset: {config.dst_zarr}')
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

    ds = xr.open_dataset(config.dst_zarr, engine='zarr', backend_kwargs=dict(consolidated=True), chunks={})

    st_yr = ds['time'].dt.year[0].item()
    en_yr = ds['time'].dt.year[-1].item()

    # varlist = ['RAINRATE', 'T2D', 'LWDOWN', 'PSFC', 'Q2D', 'SWDOWN', 'U2D', 'V2D']
    # varlist = list(ds.data_vars)
    # if 'crs' in varlist:
    #     varlist.remove('crs')

    con.print(f'Checking for missing values in {len(variables)} variables: {variables}')

    proc_info = Table(title='[bold]Years with missing values[/]')
    proc_info.add_column('Variable', justify='left', style='black')
    proc_info.add_column('Year', justify='left', style='black')
    proc_info.add_column('Max missing', justify='left', style='red')
    # proc_info.add_column('Index range', justify='left', style='grey50')

    for cvar in variables:
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
    con.save_html(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M")}_conus404_check_missing.html')


@app.command()
def create_zarr(config_file: str):
    """Create an empty daily timestep zarr store using information from a source hourly zarr

    The beginning and ending date for the newly created zarr is taken from the source hourly
    zarr at time of execution.

    :param config_file: Name of configuration file
    """

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

    ds[drop_vars].chunk(dst_chunks).to_zarr(config.dst_zarr, mode='a')
    con.print(f'Write constant variabls: {time.time() - start_time:0.3f} s')


@app.command()
def extend_time(config_file: str,
                freq: Optional[str] = '1d'):
    """Extend the time dimension in an existing daily zarr dataset

    The time extension is done by retrieving the current end time
    from the hourly zarr dataset

    :param config_file: Name of configuration file
    :param freq: frequency to use for timesteps
    """

    config = Cfg(config_file)

    src_zarr = Path(config.src_zarr).resolve()
    dst_zarr = Path(config.dst_zarr).resolve()
    dst_filename = f'{dst_zarr}/.zmetadata'

    ds_hourly = xr.open_dataset(src_zarr, engine='zarr', backend_kwargs=dict(consolidated=True), chunks={})

    # Read the consolidated metadata
    with open(dst_filename, 'r') as in_hdl:
        data = json.load(in_hdl)

    # Open the target zarr dataset
    con.print('  reading zarr store')
    ds = xr.open_dataset(dst_zarr, engine='zarr',
                         backend_kwargs=dict(consolidated=True), chunks={})

    end_date = pd.to_datetime(ds_hourly.time[-1].values).date()

    if end_date == pd.to_datetime(ds.time[-1].values).date():
        con.print(f'  [green]INFO[/]: {end_date} is already the last date in the daily zarr dataset')
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

    con.print(f'First date: {first_date.strftime("%Y-%m-%d")}, '
              f'chunk: {first_chunk} ({first_chunk_st.strftime("%Y-%m-%d")} to {first_chunk_en.strftime("%Y-%m-%d")})')
    con.print(f'Last date: {last_date.strftime("%Y-%m-%d")}, '
              f'chunk: {last_chunk} ({last_chunk_st.strftime("%Y-%m-%d")} to {last_chunk_en.strftime("%Y-%m-%d")})')

    if last_date < last_chunk_en:
        con.print(f'[green]INFO[/]: Last chunk in dataset is a partial chunk')


@app.command()
def process(config_file: str,
            chunk_index: int):
    """Convert hourly CONUS404-BA zarr dataset to daily

    :param config_file: Name of configuration file
    :param chunk_index: Index of the chunk to process
    """

    # con.print(f'HOST: {os.environ.get("HOSTNAME")}')
    # con.print(f'SLURMD_NODENAME: {os.environ.get("SLURMD_NODENAME")}')

    job_name = f'c404-ba_daily_{chunk_index}'

    config = Cfg(config_file)

    chunk_plan = config.chunk_plan

    # Amount in minutes to adjust the daily time
    adj_val = {'instant': 690,
               'cum60': 750,
               'cum_sim': 750}

    # con.print(f'dask tmp directory: {dask.config.get("temporary-directory")}')

    start_time = time.time()
    dask.config.set({"array.slicing.split_large_chunks": False})

    con.print(ptext('Settings', style='white on tan', max_width=80, expand=False))
    start_time = time.time()
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
    con.print(f'[green4]INFO[/]: Source hourly dataset: {config.src_zarr}')
    con.print(f'[green4]INFO[/]: Destination daily dataset: {config.dst_zarr}')
    con.print('='*80)

    # cluster = SLURMCluster(job_name=job_name,
    #                        queue=config.queue,
    #                        account=config.account,
    #                        interface=config.interface,
    #                        cores=config.cores_per_job,    # this is --cpus-per-task
    #                        processes=config.processes,    # this is numbers of workers for dask
    #                        memory=f'{config.memory_per_job} GiB',   # total amount of memory for job
    #                        walltime=config.walltime)
    #                        # job_cpu=8,   # this appears to override cores, but cores is still a required argument
    #                        # job_extra_directives=['--nodes=6'],
    #                        # local_directory='/home/pnorton/hytest/hourly_processing')
    #
    # con.print(cluster.job_script())
    # cluster.scale(jobs=config.max_jobs)
    #
    # client = Client(cluster)
    # client.wait_for_workers(config.processes * config.max_jobs)

    workers = 12
    threads_per = 2
    with Client(n_workers=workers, threads_per_worker=threads_per, memory_limit=None):
        # Change the default compressor to Zstd
        zarr.storage.default_compressor = Zstd(level=9)

        # con.print('--- Open source datastore ---')
        # Open hourly source datastore
        ds_hourly = xr.open_dataset(config.src_zarr, engine='zarr', backend_kwargs=dict(consolidated=True), chunks={})

        # Hourly source information needed for processing
        hrly_step_idx = 24 * chunk_plan['time']
        hrly_last_idx = ds_hourly.time.size

        # Get integration information for computing daily
        accum_types = get_accum_types(ds_hourly)
        drop_vars = accum_types['constant']

        var_list = accum_types['instantaneous']
        var_list.sort()
        # Can remove either T2D or RAINRATE to limit which variables are computed for daily
        # var_list.remove('T2D')
        con.print(f'[green4]INFO[/]: Variables available in hourly timestep dataset: {var_list}')

        # con.print(f'    --- Number of variables: {len(var_list)}')

        if chunk_index * hrly_step_idx >= hrly_last_idx:
            con.print('[red]ERROR[/]: Starting index beyond end of available hourly data')
            exit()

        proc_info = Table(title='[bold]Daily chunks processed[/]')
        proc_info.add_column('Chunk', justify='left', style='black')
        proc_info.add_column('Date range', justify='left', style='black')
        proc_info.add_column('Index range', justify='left', style='grey50')

        for cidx in range(chunk_index, chunk_index+config.num_chunks_per_job):
            loop_start = time.time()
            # print(f'--- Index {cidx:04d} ---', flush=True)

            # Get the index range for the hourly zarr store
            c_st = cidx * hrly_step_idx
            c_en = c_st + hrly_step_idx

            if c_st >= hrly_last_idx:
                con.print(f'[red]ERROR[/]: Starting index, {c_st}, is past the end of available hourly timesteps')
                break

            if c_en > hrly_last_idx:
                c_en = hrly_last_idx

            ds_daily = compute_daily(ds_hourly, var_list, st_idx=c_st, en_idx=c_en, chunks=chunk_plan)
            ds_daily.compute()

            ds_daily = adjust_time(ds_daily, time_adj=adj_val['instant'])

            ds_daily = remove_chunk_encoding(ds_daily)

            # Cumulative variables may be missing the time for the last day at the end of the POR
            # This shows as NaT in the last time index. This needs to be filled before writing
            # to the zarr store. The data values for this last day will be NaN.
            if np.isnat(ds_daily.time.values[-1]):
                ds_daily.time.values[-1] = ds_daily.time.values[-2] + np.timedelta64(1, 'D')

            # Get the index positions for inserting the chunk in the daily output zarr store
            daily_st = int(c_st / 24)
            daily_en = int(c_en / 24)
            if (daily_en - daily_st) < ds_daily.time.size:
                con.print(f'[green4]INFO[/]: Time interval changed from {daily_en - daily_st} to {ds_daily.time.size}')
                daily_en += 1

            # con.print(f'    daily range: {daily_st} ({ds_daily.time.dt.strftime("%Y-%m-%d %H:%M:%S")[0].values}) to '
            #           f'{daily_en} ({ds_daily.time.dt.strftime("%Y-%m-%d %H:%M:%S")[-1].values})'
            #           f'  timesteps: {daily_en-daily_st}')

            proc_info.add_row(f'{cidx:04d}',
                              f'{ds_daily.time.dt.strftime("%Y-%m-%d %H:%M:%S")[0].values} to '
                              f'{ds_daily.time.dt.strftime("%Y-%m-%d %H:%M:%S")[-1].values}',
                              f'{daily_st:05d} to {daily_en:05d}')
            # NOTE: Make sure the arrays that are written have the correct chunk sizes or they will be
            #       corrupted during the write.
            ds_daily.drop_vars(drop_vars, errors='ignore').to_zarr(config.dst_zarr,
                                                                   region={'time': slice(daily_st, daily_en)})

            con.print(f'[green4]INFO[/]: {cidx:04d} time: {(time.time() - loop_start) / 60.:0.3f} m')

        if len(proc_info.rows) > 0:
            con.print(proc_info)

        con.print(f'[green4]INFO[/]: Total runtime: {(time.time() - start_time) / 60.:0.3f} m')


def main():
    app()


if __name__ == '__main__':
    main()
