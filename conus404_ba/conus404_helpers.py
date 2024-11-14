import datetime
import numpy as np
import os
import pandas as pd
import rechunker
import time
import xarray as xr
import zarr


def get_accum_types(ds):
    """
    Returns dictionary of acummulation types
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


def apply_metadata(ds, rename_dims, rename_vars, remove_attrs, var_metadata):
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


def build_hourly_filelist(num_days, c_start, wrf_dir, file_pattern, verify=False):
    """
    Build a list of file paths
    """

    job_files = []

    for dd in range(num_days):
        cdate = c_start + datetime.timedelta(days=dd)

        wy_dir = f'WY{cdate.year}'
        if cdate >= datetime.datetime(cdate.year, 10, 1):
            wy_dir = f'WY{cdate.year+1}'

        for hh in range(24):
            fdate = cdate + datetime.timedelta(hours=hh)

            # 201610010000.LDASIN_DOMAIN1
            # file_pat = f'{wrf_dir}/{wy_dir}/{fdate.strftime("%Y%m%d%H%M")}.LDASIN_DOMAIN1'
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


def delete_dir(fs, path):
    """
    Recursively remove directory using fsspec
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


def read_metadata(filename):
    """
    Read the metadata information file
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


def rechunker_wrapper(source_store, target_store, temp_store, chunks=None,
                      mem=None, consolidated=False, verbose=True):

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
