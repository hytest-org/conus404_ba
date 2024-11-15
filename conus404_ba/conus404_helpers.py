import datetime
import fsspec
import numpy as np
import os
import pandas as pd
import rechunker
import time
import xarray as xr
import zarr

from typing import Dict, List, Optional, Union


def get_accum_types(ds: xr.Dataset) -> dict:
    """
    Returns dictionary of variables by acummulation types

    :param ds: Dataset of WRF model output
    :return: Dictionary of variables by accumulation type
    """

    accum_types = {}

    for vv in ds.variables:
        try:
            # accum_types[vv] = ds[vv].attrs['integration_length']
            accum_types.setdefault(ds[vv].attrs['integration_length'], [])

            if vv in ['I_ACLWDNB', 'I_ACLWUPB', 'I_ACSWDNB', 'I_ACSWDNT', 'I_ACSWUPB']:
                accum_types.setdefault('accumulated since 1979-10-01 00:00:00 bucket', [])
                accum_types['accumulated since 1979-10-01 00:00:00 bucket'].append(vv)
            else:
                accum_types.setdefault(ds[vv].attrs['integration_length'], [])
                accum_types[ds[vv].attrs['integration_length']].append(vv)
        except KeyError:
            if 'time' in ds[vv].dims:
                accum_types.setdefault('instantaneous', [])
                accum_types['instantaneous'].append(vv)
            else:
                accum_types.setdefault('constant', [])
                accum_types['constant'].append(vv)

    return accum_types


def apply_metadata(ds: xr.Dataset,
                   rename_dims: Dict,
                   rename_vars: Dict,
                   remove_attrs: List,
                   var_metadata: pd.DataFrame) -> xr.Dataset:
    """
    Update variables, dimensions, and metadata of a dataset

    :param ds: Dataset to update
    :param rename_dims: Dictionary of dimensions to rename
    :param rename_vars: Dictionary of variables to rename
    :param remove_attrs: Attributes to remove from all variables
    :param var_metadata: Dataframe of variable metadata to apply
    :return: Updated xarray dataset
    """

    avail_dims = ds.dims.keys()
    rename_dims_actual = {}

    # Only change dimensions that exist in dataset
    for kk, vv in rename_dims.items():
        if kk in avail_dims:
            rename_dims_actual[kk] = vv

    ds = ds.rename(rename_dims_actual)
    if len(rename_vars) > 0:
        ds = ds.rename_vars(rename_vars)

    # Modify the attributes
    for cvar in ds.variables:
        # Remove unneeded attributes, update the coordinates attribute
        for cattr in list(ds[cvar].attrs.keys()):
            if cattr in remove_attrs:
                del ds[cvar].attrs[cattr]

        # Apply new/updated metadata
        if cvar in var_metadata.index:
            for kk, vv in var_metadata.loc[cvar].dropna().to_dict().items():
                if kk in ['coordinates']:
                    # Add/modify encoding
                    ds[cvar].encoding.update({kk: vv})
                else:
                    ds[cvar].attrs[kk] = vv

    return ds


def build_hourly_filelist(num_days: int,
                          c_start: datetime.datetime,
                          wrf_dir: Union[str, os.PathLike],
                          file_pattern: str,
                          verify: bool = False) -> List[str]:
    """
    Build a list of file paths to model output netCDF files that match a pattern for a given number of days

    :param num_days: Number of days of model output files to include
    :param c_start: Start date for the chunk of model output files
    :param wrf_dir: Directory where the model output files are stored
    :param file_pattern: Pattern for the model output file names
    :param verify: Verify the existence of each file
    :return: List of file paths to model output netCDF files
    """

    # NOTE: The wrf_dir argument is required by the yaml config file variable file_pattern
    # NOTE: The wy_dir variable (below) is required by the yaml config file variable file_pattern

    job_files = []

    for dd in range(num_days):
        cdate = c_start + datetime.timedelta(days=dd)

        wy_dir = f'WY{cdate.year}'
        if cdate >= datetime.datetime(cdate.year, 10, 1):
            wy_dir = f'WY{cdate.year+1}'

        for hh in range(24):
            fdate = cdate + datetime.timedelta(hours=hh)

            file_pat = eval(f"f'{file_pattern}'")

            if verify:
                # Verifying the existence of each file can put a heavy load on Lustre filesystems
                # Only call this function with verify turned on when it's needed (e.g. when open_mfdataset fails).
                if os.path.exists(file_pat):
                    job_files.append(file_pat)
                else:
                    if fdate.month == 10 and fdate.day == 1 and fdate.hour == 0:
                        # Sometimes a dataset includes the first hour of the next water year in
                        # each water year directory; we need this file.
                        wy_dir = f'WY{cdate.year}'
                        file_pat = eval(f"f'{file_pattern}'")

                        if os.path.exists(file_pat):
                            job_files.append(file_pat)

                        break
            else:
                job_files.append(file_pat)
    return job_files


def delete_dir(fs: fsspec.filesystem,
               path: Union[str, os.PathLike]):
    """
    Recursively remove directory using fsspec

    :param fs: fsspec filesystem object
    :param path: Path to directory to remove
    """

    try:
        fs.rm(path, recursive=True)
    except FileNotFoundError:
        pass


def get_maxmem_per_thread(client, max_percent=0.7, verbose=False):
    """
    Returns the maximum amount of memory to use per thread for chunking.
    """

    # client: dask client
    # max_percent: Maximum percentage of memory to use for rechunking per thread

    # Max total memory in gigabytes for cluster
    total_mem = sum(vv['memory_limit'] for vv in client.scheduler_info()['workers'].values()) / 1024**3
    total_threads = sum(vv['nthreads'] for vv in client.scheduler_info()['workers'].values())

    if verbose:
        print('-'*60)
        print(f'Total memory: {total_mem:0.1f} GB; Threads: {total_threads}')

    max_mem = f'{total_mem / total_threads * max_percent:0.0f}GB'

    if verbose:
        print(f'Maximum memory per thread for rechunking: {max_mem}')
        print('-'*60)

    return max_mem


def read_metadata(filename: Union[str, os.PathLike]) -> pd.DataFrame:
    """
    Read the metadata information file

    :param filename: Path to the metadata CSV file
    :return: DataFrame of metadata information
    """

    use_cols = ['varname', 'long_name', 'standard_name', 'units', 'coordinates']

    df = pd.read_csv(filename, sep='\t', index_col='varname', usecols=use_cols)

    # Add a few empty attributes, in the future these may already exist
    # df['grid_mapping'] = ''
    # df['axis'] = ''

    # Set grid_mapping attribute for variables with non-empty coordinates attribute
    # df['grid_mapping'].mask(df['coordinates'].notnull(), 'crs', inplace=True)

    # Rechunking will crash if units for 'time' is overridden with an error
    # like the following:
    # ValueError: failed to prevent overwriting existing key units in attrs
    df.loc['time', 'units'] = np.nan

    return df


def rechunker_wrapper(source_store: xr.Dataset,
                      target_store: Union[str, os.PathLike],
                      temp_store: Union[str, os.PathLike],
                      chunks: Dict[str, int] = None,
                      mem: Optional[str] = None,
                      consolidated: bool = False,
                      verbose: bool = True):
    """Wrapper to the rechunker.rechunk() function

    :param source_store: Source store to rechunk
    :param target_store: Target store to write the rechunked data
    :param temp_store: Temporary store for rechunking
    :param chunks: Dictionary of chunk sizes for each dimension
    :param mem: Maximum memory to use for rechunking
    :param consolidated: Consolidate metadata after rechunking
    :param verbose: Print verbose output
    """

    t1 = time.time()

    if isinstance(source_store, xr.Dataset):
        g = source_store  # Work directly with a dataset
        ds_chunk = g
    else:
        g = zarr.group(str(source_store))
        # Get the correct shape from loading the store as xr.dataset and parse the chunks
        ds_chunk = xr.open_zarr(str(source_store))

    group_chunks = {}

    # Newer tuple version that also takes into account when specified chunks are larger than the array
    for var in ds_chunk.variables:
        # Pick appropriate chunks from above, and default to full length chunks for
        # dimensions that are not in `chunks` above.
        group_chunks[var] = []

        for di in ds_chunk[var].dims:
            if di in chunks.keys():
                if chunks[di] > len(ds_chunk[di]):
                    group_chunks[var].append(len(ds_chunk[di]))
                else:
                    group_chunks[var].append(chunks[di])

            else:
                group_chunks[var].append(len(ds_chunk[di]))

        group_chunks[var] = tuple(group_chunks[var])

    if verbose:
        print(f"Rechunking to: {group_chunks}")
        print(f"Memory: {mem}")

    rechunked = rechunker.rechunk(g, target_chunks=group_chunks, max_mem=mem,
                                  target_store=target_store, temp_store=temp_store)
    rechunked.execute(retries=10)

    if consolidated:
        if verbose:
            print('Consolidating metadata')

        zarr.convenience.consolidate_metadata(target_store)

    if verbose:
        print(f'    rechunker: {time.time() - t1:0.3f} s')
