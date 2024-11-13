#!/usr/bin/env python3

from cyclopts import App, Parameter   # , validators
from pathlib import Path

import dask
import json
import numpy as np
import os
import pandas as pd
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

from typing import Dict, List, Optional
from zarr.convenience import consolidate_metadata
from zarr.util import NumberEncoder

from ..conus404_helpers import get_accum_types
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


def compute_daily(ds: xr.Dataset,
                  var_list: List,
                  st_idx: int,
                  en_idx: int,
                  chunks: Optional[Dict[str, int]] = None):
    """Compute daily values from a source hourly dataset

    :param ds: Source hourly dataset
    :param var_list: List of variables to process
    :param st_idx: Starting index in source hourly dataset
    :param en_idx: Ending index in source hourly dataset
    :param chunks: Dictionary containing chunk sizes
    """

    if chunks is None:
        chunks = {}

    # NOTE: make sure the arrays have the correct final chunking.

    ds_day_cnk = ds[var_list].isel(time=slice(st_idx, en_idx))
    con.print(f'    instant: hourly range: {st_idx} ({ds_day_cnk.time.dt.strftime("%Y-%m-%d %H:%M:%S")[0].values}) to '
              f'{en_idx} ({ds_day_cnk.time.dt.strftime("%Y-%m-%d %H:%M:%S")[-1].values})')

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


def adjust_time(ds: xr.Dataset, time_adj: int):
    """Adjust time values after computing daily from hourly to align time boundaries

    :param ds: Dataset to adjust time values
    :param time_adj: Number of seconds to adjust
    """

    # Adjust the time values, pass the original encoding to the new time index
    save_enc = ds.time.encoding
    del save_enc['chunks']

    ds['time'] = ds['time'] - np.timedelta64(time_adj, 'm')
    ds.time.encoding = save_enc

    return ds


def remove_chunk_encoding(ds: xr.Dataset):
    """Remove existing encoding from variables in dataset

    :param ds: Dataset to remove encoding from
    """

    # Remove the existing encoding for chunks
    for vv in ds.variables:
        try:
            del ds[vv].encoding['chunks']
        except KeyError:
            pass

    return ds


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
    """Extend the time dimension in an existing zarr dataset

    The time extension is done by retrieving the current ending time
    from the hourly zarr dataset

    :param config_file: Name of configuration file
    :param freq: frequency to use for timesteps
    """

    config = Cfg(config_file)

    src_zarr = Path(config.src_zarr).resolve()
    dst_zarr = Path(config.dst_zarr).resolve()
    # end_date = config.end_date
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

    ds.coords['time'] = dates
    ds['time'].encoding.update(save_enc)

    ds[['time']].to_zarr(dst_zarr, mode='a')


@app.command()
def process(config_file: str,
            chunk_index: int):
    """Convert hourly CONUS404-BA zarr dataset to daily

    :param config_file: Name of configuration file
    :param chunk_index: Index of the chunk to process
    """

    con.print(f'HOST: {os.environ.get("HOSTNAME")}')
    con.print(f'SLURMD_NODENAME: {os.environ.get("SLURMD_NODENAME")}')

    job_name = f'c404BA_daily_{chunk_index}'

    config = Cfg(config_file)

    chunk_plan = config.chunk_plan

    # Amount in minutes to adjust the daily time
    adj_val = {'instant': 690,
               'cum60': 750,
               'cum_sim': 750}

    con.print(f'dask tmp directory: {dask.config.get("temporary-directory")}')

    start_time = time.time()
    dask.config.set({"array.slicing.split_large_chunks": False})

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

    # Change the default compressor to Zstd
    zarr.storage.default_compressor = Zstd(level=9)

    con.print('--- Open source datastore ---')
    # Open hourly source datastore
    ds_hourly = xr.open_dataset(config.src_zarr, engine='zarr', backend_kwargs=dict(consolidated=True), chunks={})

    # Hourly source information needed for processing
    # hrly_days_per_cnk = 6
    # hrly_time_cnk = 24 * hrly_days_per_cnk
    hrly_step_idx = 24 * chunk_plan['time']
    hrly_last_idx = ds_hourly.time.size

    # Get integration information for computing daily
    accum_types = get_accum_types(ds_hourly)
    drop_vars = accum_types['constant']

    var_list = accum_types['instantaneous']
    var_list.sort()
    # Can remove either T2D or RAINRATE to limit which variables are computed for daily
    # var_list.remove('T2D')

    con.print(f'    --- Number of variables: {len(var_list)}')

    if chunk_index * hrly_step_idx >= hrly_last_idx:
        con.print('[red]ERROR[/]: Starting index beyond end of available hourly data')
        exit()

    for cidx in range(chunk_index, chunk_index+config.num_chunks_per_job):
        # for c_loop in range(args.loop):
        loop_start = time.time()
        print(f'--- Index {cidx:04d} ---', flush=True)

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
            con.print(f'    time interval changed from {daily_en - daily_st} to {ds_daily.time.size}')
            daily_en += 1

        con.print(f'    daily range: {daily_st} ({ds_daily.time.dt.strftime("%Y-%m-%d %H:%M:%S")[0].values}) to '
                  f'{daily_en} ({ds_daily.time.dt.strftime("%Y-%m-%d %H:%M:%S")[-1].values})'
                  f'  timesteps: {daily_en-daily_st}')

        # print('    --- write to zarr store', flush=True)
        # NOTE: Make sure the arrays that are written have the correct chunk sizes or they will be
        #       corrupted during the write.
        ds_daily.drop_vars(drop_vars, errors='ignore').to_zarr(config.dst_zarr,
                                                               region={'time': slice(daily_st, daily_en)})

        con.print(f'    time: {(time.time() - loop_start) / 60.:0.3f} m')

    con.print(f'Runtime: {(time.time() - start_time) / 60.:0.3f} m')


def main():
    app()


if __name__ == '__main__':
    main()
